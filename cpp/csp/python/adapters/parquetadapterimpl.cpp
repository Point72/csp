#include <numpy/ndarrayobject.h>
#include <csp/adapters/parquet/ParquetInputAdapterManager.h>
#include <csp/adapters/parquet/ParquetOutputAdapterManager.h>
#include <csp/adapters/parquet/ParquetDictBasketOutputWriter.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>
#include <csp/engine/CppNode.h>
#include <csp/engine/CspType.h>
#include <csp/python/PyCppNode.h>
#include <csp/python/PyNodeWrapper.h>
#include <csp/python/NumpyConversions.h>
#include <csp/python/adapters/ArrowNumpyListReader.h>
#include <csp/python/adapters/ArrowNumpyListWriter.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/io/file.h>
#include <parquet/arrow/reader.h>
#include <parquet/file_reader.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <filesystem>
#include <locale>
#include <codecvt>
#include <numeric>

using namespace csp::adapters::parquet;

namespace csp::cppnodes
{
DECLARE_CPPNODE( parquet_dict_basket_writer )
{
    SCALAR_INPUT( std::string, column_name );
    SCALAR_INPUT( DialectGenericType, writer );
    TS_DICTBASKET_INPUT( Generic, input );
    TS_INPUT( std::string, filename_provider );


    STATE_VAR( csp::adapters::parquet::ParquetDictBasketOutputWriter*, s_outputWriter );

    INIT_CPPNODE( parquet_dict_basket_writer )
    {
        const csp::python::PyObjectPtr *writerObjectPtr = reinterpret_cast<const csp::python::PyObjectPtr *>(&writer);

        auto managerObjectPtr = csp::python::PyObjectPtr::incref(
                PyObject_CallMethod( writerObjectPtr -> get(), "_get_output_adapter_manager", "" ) );

        auto *outputAdapterManager = dynamic_cast<ParquetOutputAdapterManager *>( csp::python::PyAdapterManagerWrapper::extractAdapterManager(
                managerObjectPtr.get() ));

        s_outputWriter = outputAdapterManager -> createDictOutputBasketWriter( column_name.value().c_str(), input.type() );
    }

    INVOKE()
    {
        if( filename_provider.ticked() ) [[unlikely]]
        {
            s_outputWriter -> onFileNameChange( filename_provider.lastValue() );
        }
        if( s_outputWriter -> isFileOpen() )
        {
            const auto &shape          = input.shape();
            for( auto  &&tickedInputIt = input.tickedinputs(); tickedInputIt; ++tickedInputIt )
            {
                s_outputWriter -> writeValue( shape[ tickedInputIt.elemId() ], tickedInputIt.get() );
            }
        }
    }
};

EXPORT_CPPNODE( parquet_dict_basket_writer );
}


REGISTER_CPPNODE( csp::cppnodes, parquet_dict_basket_writer );

namespace
{

// Wraps a Python "stream factory" callable.
// Supports non-split (yields {col: reader}) and split-column protocols.
class PyRecordBatchStreamSource : public csp::adapters::parquet::RecordBatchStreamSource
{
public:
    PyRecordBatchStreamSource( PyObject *factory )
            : m_factory( csp::python::PyObjectPtr::incref( factory ) )
    {
    }

    void init( csp::DateTime start, csp::DateTime end,
               const std::set<std::string> & neededColumns ) override
    {
        auto tp = csp::python::PyObjectPtr::own( PyTuple_New( 3 ) );
        if( !tp.get() )
            CSP_THROW( csp::python::PythonPassthrough, "" );

        PyTuple_SET_ITEM( tp.get(), 0, csp::python::toPython( start ) );
        PyTuple_SET_ITEM( tp.get(), 1, csp::python::toPython( end ) );

        if( neededColumns.empty() )
        {
            Py_INCREF( Py_None );
            PyTuple_SET_ITEM( tp.get(), 2, Py_None );
        }
        else
        {
            auto pyList = csp::python::PyObjectPtr::own( PyList_New( neededColumns.size() ) );
            if( !pyList.get() )
                CSP_THROW( csp::python::PythonPassthrough, "" );
            Py_ssize_t idx = 0;
            for( auto & col : neededColumns )
                PyList_SET_ITEM( pyList.get(), idx++, PyUnicode_FromStringAndSize( col.c_str(), col.size() ) );
            PyTuple_SET_ITEM( tp.get(), 2, pyList.release() );
        }

        m_iter = csp::python::PyObjectPtr::check( PyObject_Call( m_factory.ptr(), tp.get(), nullptr ) );
        CSP_TRUE_OR_THROW( PyIter_Check( m_iter.ptr() ), csp::TypeError,
                           "Stream factory expected to return iterator" );
    }

    bool nextStream() override
    {
        if( m_iter.ptr() == nullptr )
            return false;

        auto nextVal = csp::python::PyObjectPtr::own( PyIter_Next( m_iter.ptr() ) );
        if( PyErr_Occurred() )
            CSP_THROW( csp::python::PythonPassthrough, "" );
        if( nextVal.get() == nullptr )
            return false;

        // Expect a dict of {col_name: RecordBatchReader}
        CSP_TRUE_OR_THROW( PyDict_Check( nextVal.get() ), csp::TypeError,
                           "Stream factory expected to yield {column: reader} dicts" );

        importColumnReaderDict( nextVal.get(), m_columnReaders );
        return true;
    }

    const ColumnReaderMap & columnReaders() const override
    {
        return m_columnReaders;
    }

private:
    static std::shared_ptr<::arrow::RecordBatchReader> importRecordBatchReader( PyObject *pyReader )
    {
        auto capsule = csp::python::PyObjectPtr::own(
            PyObject_CallMethod( pyReader, "__arrow_c_stream__", nullptr ) );
        if( !capsule.get() || PyErr_Occurred() )
            CSP_THROW( csp::python::PythonPassthrough, "" );

        auto *stream = reinterpret_cast<struct ArrowArrayStream *>(
            PyCapsule_GetPointer( capsule.get(), "arrow_array_stream" ) );
        if( !stream )
            CSP_THROW( csp::ValueError, "Failed to get ArrowArrayStream from PyCapsule" );

        auto result = ::arrow::ImportRecordBatchReader( stream );
        if( !result.ok() )
            CSP_THROW( csp::ValueError, "Failed to import RecordBatchReader: " << result.status().ToString() );

        return result.ValueUnsafe();
    }

    static void importColumnReaderDict( PyObject *pyDict, ColumnReaderMap & out )
    {
        out.clear();
        CSP_TRUE_OR_THROW( PyDict_Check( pyDict ), csp::TypeError,
                           "Expected dict of column readers" );

        PyObject *key, *val;
        Py_ssize_t pos = 0;
        while( PyDict_Next( pyDict, &pos, &key, &val ) )
        {
            const char *colName = PyUnicode_AsUTF8( key );
            if( !colName )
                CSP_THROW( csp::python::PythonPassthrough, "" );
            out[ colName ] = importRecordBatchReader( val );
        }
    }

    csp::python::PyObjectPtr m_factory;
    csp::python::PyObjectPtr m_iter;

    ColumnReaderMap m_columnReaders;
};

// Native C++ parquet reader — opens parquet files directly, bypassing Python.
// Used for regular and split-column parquet. IPC/memory tables use PyRecordBatchStreamSource.
class NativeParquetStreamSource : public csp::adapters::parquet::RecordBatchStreamSource
{
public:
    NativeParquetStreamSource( PyObject * filenameGen, bool splitColumns, bool allowMissingFiles )
        : m_filenameGen( csp::python::PyObjectPtr::incref( filenameGen ) )
        , m_splitColumns( splitColumns )
        , m_allowMissingFiles( allowMissingFiles )
    {
    }

    void init( csp::DateTime start, csp::DateTime end,
               const std::set<std::string> & neededColumns ) override
    {
        m_neededColumns = neededColumns;
        m_filenames.clear();
        m_fileIdx = 0;

        // Call Python filename generator(starttime, endtime)
        auto pyStart = csp::python::PyObjectPtr::own( csp::python::toPython( start ) );
        auto pyEnd   = csp::python::PyObjectPtr::own( csp::python::toPython( end ) );

        auto pyResult = csp::python::PyObjectPtr::own(
            PyObject_CallFunctionObjArgs( m_filenameGen.ptr(), pyStart.get(), pyEnd.get(), nullptr ) );
        if( !pyResult.get() || PyErr_Occurred() )
            CSP_THROW( csp::python::PythonPassthrough, "" );

        // Convert iterable to iterator
        auto pyIter = csp::python::PyObjectPtr::own( PyObject_GetIter( pyResult.get() ) );
        if( !pyIter.get() || PyErr_Occurred() )
            CSP_THROW( csp::python::PythonPassthrough, "" );

        while( true )
        {
            auto item = csp::python::PyObjectPtr::own( PyIter_Next( pyIter.ptr() ) );
            if( PyErr_Occurred() )
                CSP_THROW( csp::python::PythonPassthrough, "" );
            if( !item.get() )
                break;
            const char * s = PyUnicode_AsUTF8( item.get() );
            if( !s )
                CSP_THROW( csp::python::PythonPassthrough, "" );
            m_filenames.emplace_back( s );
        }
    }

    bool nextStream() override
    {
        m_columnReaders.clear();
        // Release previous stream's FileReaders
        m_prevFileReaders = std::move( m_fileReaders );
        m_fileReaders.clear();

        while( m_fileIdx < m_filenames.size() )
        {
            const auto & path = m_filenames[m_fileIdx++];

            if( m_splitColumns )
            {
                if( !std::filesystem::is_directory( path ) )
                    CSP_THROW( csp::ValueError,
                               "split_columns_to_files expects a directory, got: " << path );
                return openSplitDirectory( path );
            }
            else
            {
                if( !std::filesystem::exists( path ) )
                {
                    if( m_allowMissingFiles )
                        continue;
                    CSP_THROW( csp::ValueError, "Parquet file not found: " << path );
                }
                return openSingleFile( path );
            }
        }
        return false;
    }

    const ColumnReaderMap & columnReaders() const override
    {
        return m_columnReaders;
    }

private:
    // Open a single parquet file, projecting only needed columns.
    bool openSingleFile( const std::string & path )    {
        auto fileReader = makeFileReader( path );
        if( !fileReader )
            return false;

        std::vector<int> colIndices;
        if( !m_neededColumns.empty() )
        {
            std::shared_ptr<::arrow::Schema> arrowSchema;
            auto status = fileReader -> GetSchema( &arrowSchema );
            if( status.ok() && arrowSchema )
            {
                for( int i = 0; i < arrowSchema -> num_fields(); ++i )
                {
                    auto fieldName = arrowSchema -> field( i ) -> name();
                    if( m_neededColumns.count( fieldName ) )
                    {
                        colIndices.push_back( i );
                        continue;
                    }
                    // Include parent fields of needed nested columns (e.g. "struct.i" → "struct")
                    std::string prefix = fieldName + ".";
                    auto lb = m_neededColumns.lower_bound( prefix );
                    if( lb != m_neededColumns.end() && lb -> compare( 0, prefix.size(), prefix ) == 0 )
                        colIndices.push_back( i );
                }
            }
        }

        auto reader = getRecordBatchReader( fileReader, colIndices );
        if( !reader )
            return false;

        m_columnReaders[ reader -> schema() -> field( 0 ) -> name() ] = reader;
        m_fileReaders.push_back( std::move( fileReader ) );
        return true;
    }

    // Open a split-column directory: each .parquet file is one column.
    bool openSplitDirectory( const std::string & dirPath )    {
        // Collect column files, sorted
        std::map<std::string, std::string> colFiles;
        for( auto & entry : std::filesystem::directory_iterator( dirPath ) )
        {
            if( entry.path().extension() != ".parquet" )
                continue;
            auto colName = entry.path().stem().string();
            if( !m_neededColumns.empty() && !m_neededColumns.count( colName ) )
                continue;
            colFiles[colName] = entry.path().string();
        }

        if( colFiles.empty() )
            return false;

        for( auto & [colName, filePath] : colFiles )
        {
            auto fileReader = makeFileReader( filePath );
            if( !fileReader )
                continue;

            auto reader = getRecordBatchReader( fileReader, {} );
            if( !reader )
                continue;

            m_columnReaders[colName] = reader;
            m_fileReaders.push_back( std::move( fileReader ) );
        }

        return !m_columnReaders.empty();
    }

    static std::unique_ptr<::parquet::arrow::FileReader> makeFileReader( const std::string & path )
    {
        auto fileResult = ::arrow::io::ReadableFile::Open( path );
        if( !fileResult.ok() )
            CSP_THROW( csp::ValueError, "Failed to open " << path << ": " << fileResult.status().ToString() );

        auto parquetReader = ::parquet::ParquetFileReader::Open( fileResult.ValueUnsafe() );

        std::unique_ptr<::parquet::arrow::FileReader> fileReader;
        auto status = ::parquet::arrow::FileReader::Make(
            ::arrow::default_memory_pool(), std::move( parquetReader ), &fileReader );
        if( !status.ok() )
            CSP_THROW( csp::ValueError, "Failed to create Arrow FileReader for " << path << ": " << status.ToString() );

        return fileReader;
    }

    static std::shared_ptr<::arrow::RecordBatchReader> getRecordBatchReader(
        const std::unique_ptr<::parquet::arrow::FileReader> & fileReader,
        const std::vector<int> & colIndices )
    {
        int numRG = fileReader -> num_row_groups();
        std::vector<int> rowGroups( numRG );
        std::iota( rowGroups.begin(), rowGroups.end(), 0 );

        ::arrow::Result<std::unique_ptr<::arrow::RecordBatchReader>> result;
        if( colIndices.empty() )
            result = fileReader -> GetRecordBatchReader( rowGroups );
        else
            result = fileReader -> GetRecordBatchReader( rowGroups, colIndices );

        if( !result.ok() )
            CSP_THROW( csp::ValueError, "GetRecordBatchReader failed: " << result.status().ToString() );

        // Convert unique_ptr → shared_ptr
        return std::shared_ptr<::arrow::RecordBatchReader>( std::move( result ).ValueUnsafe() );
    }

    csp::python::PyObjectPtr                                      m_filenameGen;
    bool                                                          m_splitColumns;
    bool                                                          m_allowMissingFiles;
    std::set<std::string>                                         m_neededColumns;
    std::vector<std::string>                                      m_filenames;
    size_t                                                        m_fileIdx = 0;
    ColumnReaderMap                                                m_columnReaders;
    // FileReaders must outlive their RecordBatchReaders
    std::vector<std::unique_ptr<::parquet::arrow::FileReader>>    m_fileReaders;
    std::vector<std::unique_ptr<::parquet::arrow::FileReader>>    m_prevFileReaders;
};

}

namespace csp::python
{

csp::AdapterManager *create_parquet_input_adapter_manager_impl( PyEngine *engine, const Dictionary &properties,
                                                                ParquetInputAdapterManager::RecordBatchStreamSourcePtr streamSource )
{
    auto res = engine -> engine() -> createOwnedObject<ParquetInputAdapterManager>( properties, streamSource );
    return res;
}

static InputAdapter *
create_parquet_input_adapter( csp::AdapterManager *manager, PyEngine *pyengine, PyObject *pyType, PushMode pushMode,
                              PyObject *args )
{
    auto &cspType = pyTypeAsCspType( pyType );

    PyObject *pyProperties;
    PyObject *type;

    auto *parquetManager = dynamic_cast<ParquetInputAdapterManager *>( manager );
    if( !parquetManager )
        CSP_THROW( TypeError, "Expected ParquetAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &type,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    auto propertiesDict = fromPython<Dictionary>( pyProperties );

    if( propertiesDict.get( "is_array", false ) )
    {
        // Array fields use DIALECT_GENERIC type — the registered ListFieldReaderFactory
        // handles actual numpy array reading via ColumnDispatcher/FieldReader.
        auto dialectGenericType = CspType::DIALECT_GENERIC();
        return parquetManager -> getInputAdapter( dialectGenericType, propertiesDict, pushMode );
    }
    else
    {
        return parquetManager -> getInputAdapter( cspType, propertiesDict, pushMode );
    }
}

template< typename CspCType >
class NumpyArrayWriterImpl : public TypedDialectGenericListWriterInterface<CspCType>
{
public:
    NumpyArrayWriterImpl( PyArray_Descr *expectedArrayDesc )
            : m_expectedArrayDesc( expectedArrayDesc )
    {
    }

    void writeItems( const csp::DialectGenericType &listObject ) override
    {
        PyObject *object = csp::python::toPythonBorrowed( listObject );
        if( !PyArray_Check( object ) )
            CSP_THROW( csp::TypeError, "While writing to parquet expected numpy array type, got " << Py_TYPE( object ) -> tp_name );

        PyArrayObject *arrayObject = ( PyArrayObject * ) ( object );
        char npy_type = PyArray_DESCR( arrayObject ) -> type;
        if( PyArray_DESCR( arrayObject ) -> kind != m_expectedArrayDesc -> kind )
            CSP_THROW( csp::TypeError,
                       "Expected array of type " << csp::python::PyObjectPtr::own( PyObject_Repr( ( PyObject * ) m_expectedArrayDesc ) )
                                                 << " got "
                                                 << csp::python::PyObjectPtr::own( PyObject_Repr( ( PyObject * ) PyArray_DESCR( arrayObject ) ) ) );

        auto ndim = PyArray_NDIM( arrayObject );
        CSP_TRUE_OR_THROW_RUNTIME( ndim == 1, "While writing to parquet expected numpy array with 1 dimension" << " got " << ndim );
        switch( npy_type )
        {
            case NPY_BYTELTR:      writeValues<char>( arrayObject );            break;
            case NPY_UBYTELTR:     writeValues<unsigned char>( arrayObject );   break;
            case NPY_SHORTLTR:     writeValues<short>( arrayObject );           break;
            case NPY_USHORTLTR:    writeValues<unsigned short>( arrayObject );  break;
            case NPY_INTLTR:       writeValues<int>( arrayObject );             break;
            case NPY_UINTLTR:      writeValues<unsigned int>( arrayObject );    break;
            case NPY_LONGLTR:      writeValues<long>( arrayObject );            break;
            case NPY_ULONGLTR:     writeValues<unsigned long>( arrayObject );   break;
            case NPY_LONGLONGLTR:  writeValues<long long>( arrayObject );       break;
            case NPY_ULONGLONGLTR: writeValues<unsigned long long>( arrayObject ); break;
            case NPY_FLOATLTR:  writeValues<float>( arrayObject );  break;
            case NPY_DOUBLELTR: writeValues<double>( arrayObject ); break;
            default:
                writeValues<CspCType>( arrayObject );
        }
    }
private:
    template<typename NumpyCType>
    void writeValues( PyArrayObject * arrayObject )
    {
        auto arraySize = PyArray_Size( ( PyObject * ) arrayObject );
        if( PyArray_ISCARRAY_RO(arrayObject) )
        {
            NumpyCType* data = reinterpret_cast<NumpyCType*>( PyArray_DATA( arrayObject ) );
            for (decltype(arraySize) i = 0; i < arraySize; ++i)
                this->writeValue(static_cast<CspCType>(data[i]));
        }
        else
        {
            for (decltype(arraySize) i = 0; i < arraySize; ++i)
                this->writeValue(static_cast<CspCType>(*reinterpret_cast<NumpyCType*>(PyArray_GETPTR1(arrayObject, i))));
        }
    }

    PyArray_Descr *m_expectedArrayDesc;
};

class NumpyUnicodeArrayWriter : public TypedDialectGenericListWriterInterface<std::string>
{
public:
    void writeItems( const csp::DialectGenericType &listObject ) override
    {
        PyObject *object = csp::python::toPythonBorrowed( listObject );
        if( !PyArray_Check( object ) )
            CSP_THROW( csp::TypeError, "While writing to parquet expected numpy array type, got " << Py_TYPE( object ) -> tp_name );

        PyArrayObject *arrayObject = ( PyArrayObject * ) ( object );
        if( PyArray_DESCR( arrayObject ) -> type_num != NPY_UNICODE )
            CSP_THROW( csp::TypeError, "Expected unicode array, got " <<
                       csp::python::PyObjectPtr::own( PyObject_Repr( ( PyObject * ) PyArray_DESCR( arrayObject ) ) ) );

        auto elementSize = PyDataType_ELSIZE( PyArray_DESCR( arrayObject ) );
        auto ndim        = PyArray_NDIM( arrayObject );
        CSP_TRUE_OR_THROW_RUNTIME( ndim == 1, "While writing to parquet expected numpy array with 1 dimension" << " got " << ndim );
        std::wstring_convert<std::codecvt_utf8<char32_t>,char32_t> converter;

        auto arraySize = PyArray_Size( object );
        if( PyArray_ISCARRAY_RO( arrayObject ) )
        {
            auto data = reinterpret_cast<char *>(PyArray_DATA( arrayObject ));
            for( decltype( arraySize ) i = 0; i < arraySize; ++i )
            {
                std::string value = converter.to_bytes( reinterpret_cast<char32_t*>(data + elementSize * i),
                                                        reinterpret_cast<char32_t*>(data + elementSize * ( i + 1 )) );
                this -> writeValue( value );
            }
        }
        else
        {
            for( decltype( arraySize ) i = 0; i < arraySize; ++i )
            {
                char        *elementPtr = reinterpret_cast<char *>(PyArray_GETPTR1( arrayObject, i ));
                std::string value       = converter.to_bytes( reinterpret_cast<char32_t*>(elementPtr),
                                                              reinterpret_cast<char32_t*>(elementPtr + elementSize ) );
                this -> writeValue( value );
            }
        }
    }
};

static inline DialectGenericListWriterInterface::Ptr create_numpy_array_writer_impl( const csp::CspTypePtr &type )
{
    try
    {
        return csp::PartialSwitchCspType<csp::CspType::Type::DOUBLE, csp::CspType::Type::INT64,
                csp::CspType::Type::BOOL, csp::CspType::Type::STRING>::invoke(
                type.get(),
                []( auto tag ) -> DialectGenericListWriterInterface::Ptr
                {
                    using CValueType = typename decltype( tag )::type;
                    auto numpy_dtype = PyArray_DescrFromType( csp::python::NPY_TYPE<CValueType>::value );
                    if constexpr (std::is_same_v<CValueType,std::string>)
                        return std::make_shared<NumpyUnicodeArrayWriter>();
                    else
                        return std::make_shared<NumpyArrayWriterImpl<CValueType>>(numpy_dtype);
                }
        );
    }
    catch( csp::TypeError &e )
    {
        CSP_THROW( csp::TypeError, "Unsupported array value type when writing to parquet:" << type -> type().asString() );
    }
}

static OutputAdapter *create_parquet_output_adapter( csp::AdapterManager *manager, PyEngine *pyengine, PyObject *args )
{
    PyObject *pyProperties;
    PyObject *pyType;

    auto *parquetManager = dynamic_cast<ParquetOutputAdapterManager *>( manager );
    if( !parquetManager )
        CSP_THROW( TypeError, "Expected ParquetAdapterManager" );

    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyType_Type, &pyType,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );

    auto &cspType = pyTypeAsCspType( pyType );
    auto &&propertiesDict = fromPython<Dictionary>( pyProperties );
    if( propertiesDict.get( "is_array", false ) )
    {
        auto &&valueType = pyTypeAsCspType( toPythonBorrowed( propertiesDict.get<DialectGenericType>( "array_value_type" ) ) );
        return parquetManager -> getListOutputAdapter( valueType, propertiesDict, create_numpy_array_writer_impl( valueType ) );
    }
    else
    {
        return parquetManager -> getOutputAdapter( cspType, propertiesDict );
    }
}

static OutputAdapter *create_parquet_dict_basket_output_adapter( csp::AdapterManager *manager, PyEngine *pyengine, PyObject *args )
{
    PyObject *pyProperties;
    PyObject *pyType;
    auto     *parquetManager = dynamic_cast<ParquetOutputAdapterManager *>( manager );
    if( !parquetManager )
        CSP_THROW( TypeError, "Expected ParquetOutputAdapterManager" );
    if( !PyArg_ParseTuple( args, "O!O!",
                           &PyTuple_Type, &pyType,
                           &PyDict_Type, &pyProperties ) )
        CSP_THROW( PythonPassthrough, "" );
    PyObject *keyType;
    PyObject *valueType;
    if( !PyArg_ParseTuple( pyType, "O!O!",
                           &PyType_Type, &keyType,
                           &PyType_Type, &valueType ) )
        CSP_THROW( PythonPassthrough, "Invalid basket key/value tuple" );

    auto cspKeyType   = pyTypeAsCspType( keyType );
    auto cspValueType = pyTypeAsCspType( valueType );

    CSP_THROW( NotImplemented, "Output basket is not implement yet" );
}

static OutputAdapter *parquet_output_filename_adapter( csp::AdapterManager *manager, PyEngine *pyengine, PyObject *args )
{
    auto *parquetManager = dynamic_cast<ParquetOutputAdapterManager *>( manager );
    if( !parquetManager )
        CSP_THROW( TypeError, "Expected ParquetAdapterManager" );

    if( !PyArg_ParseTuple( args, "" ) )
        CSP_THROW( PythonPassthrough, "" );

    return parquetManager -> createOutputFileNameAdapter();
}

static PyObject *create_parquet_input_adapter_manager( PyObject *args )
{
    CSP_BEGIN_METHOD ;
        PyEngine *pyEngine        = nullptr;
        PyObject *pyProperties    = nullptr;
        PyObject *pyStreamFactory = nullptr;

        if( !PyArg_ParseTuple( args, "O!O!O",
                               &PyEngine::PyType, &pyEngine,
                               &PyDict_Type, &pyProperties,
                               &pyStreamFactory ) )
            CSP_THROW( PythonPassthrough, "" );

        if( !PyCallable_Check( pyStreamFactory ) )
            CSP_THROW( csp::TypeError, "stream_factory must be callable" );

        auto streamSource = std::make_shared<PyRecordBatchStreamSource>( pyStreamFactory );
        auto *adapterMgr = create_parquet_input_adapter_manager_impl( pyEngine, fromPython<Dictionary>( pyProperties ),
                                                                       streamSource );
        auto res         = PyCapsule_New( adapterMgr, "adapterMgr", nullptr );
        return res;
    CSP_RETURN_NULL;
}

// Native parquet path — C++ opens files directly.
static PyObject *create_parquet_input_adapter_manager_native( PyObject *args )
{
    CSP_BEGIN_METHOD ;
        PyEngine *pyEngine        = nullptr;
        PyObject *pyProperties    = nullptr;
        PyObject *pyFilenameGen   = nullptr;

        if( !PyArg_ParseTuple( args, "O!O!O",
                               &PyEngine::PyType, &pyEngine,
                               &PyDict_Type, &pyProperties,
                               &pyFilenameGen ) )
            CSP_THROW( PythonPassthrough, "" );

        if( !PyCallable_Check( pyFilenameGen ) )
            CSP_THROW( csp::TypeError, "filename generator must be callable" );

        auto properties     = fromPython<Dictionary>( pyProperties );
        bool splitColumns   = properties.get( "split_columns_to_files", false );
        bool allowMissing   = properties.get( "allow_missing_files", false );

        auto streamSource = std::make_shared<NativeParquetStreamSource>( pyFilenameGen, splitColumns, allowMissing );
        auto *adapterMgr = create_parquet_input_adapter_manager_impl( pyEngine, properties, streamSource );
        auto res         = PyCapsule_New( adapterMgr, "adapterMgr", nullptr );
        return res;
    CSP_RETURN_NULL;
}

csp::AdapterManager *create_parquet_output_adapter_manager( PyEngine *engine, const Dictionary &properties )
{
    ParquetOutputAdapterManager::FileVisitorCallback fileVisitor;
    DialectGenericType pyFilenameVisitorDG;
    if( properties.tryGet( "file_visitor", pyFilenameVisitorDG ) )
    {
        PyObjectPtr pyFilenameVisitor = PyObjectPtr::own( toPython( pyFilenameVisitorDG ) );
        fileVisitor = [pyFilenameVisitor]( const std::string & filename )
            {
                PyObjectPtr rv =  PyObjectPtr::own( PyObject_CallFunction( pyFilenameVisitor.get(), "O", PyObjectPtr::own( toPython( filename ) ).get() ) );
                if( !rv.get() )
                    CSP_THROW( PythonPassthrough, "" );
            };
    }
    return engine -> engine() -> createOwnedObject<ParquetOutputAdapterManager>( properties, fileVisitor );
}


REGISTER_ADAPTER_MANAGER_CUSTOM_CREATOR( _parquet_input_adapter_manager, create_parquet_input_adapter_manager );
REGISTER_ADAPTER_MANAGER_CUSTOM_CREATOR( _parquet_input_adapter_manager_native, create_parquet_input_adapter_manager_native );

REGISTER_ADAPTER_MANAGER( _parquet_output_adapter_manager, create_parquet_output_adapter_manager );

REGISTER_INPUT_ADAPTER( _parquet_input_adapter, create_parquet_input_adapter );

REGISTER_OUTPUT_ADAPTER( _parquet_output_adapter, create_parquet_output_adapter );

REGISTER_OUTPUT_ADAPTER( _parquet_dict_basket_output_adapter, create_parquet_dict_basket_output_adapter );

REGISTER_OUTPUT_ADAPTER( _parquet_output_filename_adapter, parquet_output_filename_adapter );
static PyModuleDef _parquetadapterimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_parquetadapterimpl",
        "_parquetadapterimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__parquetadapterimpl( void )
{
    PyObject *m;
    
    m = PyModule_Create( &_parquetadapterimpl_module );

    if( m == NULL )
    {
        return NULL;
    }

    if( !InitHelper::instance().execute( m ) )
    {
        return NULL;
    }

    import_array();

    csp::python::registerNumpyListFieldReaderFactory();
    csp::python::registerNumpyListFieldWriterFactory();

    return m;
}

}
