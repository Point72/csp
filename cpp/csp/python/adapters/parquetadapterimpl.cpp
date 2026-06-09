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
#include <arrow/util/future.h>
#include <arrow/util/thread_pool.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/schema.h>
#include <parquet/file_reader.h>
#include <parquet/properties.h>
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

// Wraps a Python "stream factory" callable (yields {col: reader} dicts).
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

        CSP_TRUE_OR_THROW( PyDict_Check( nextVal.get() ), csp::TypeError,
                           "Stream factory expected to yield {column: reader} dicts" );

        importColumnSourceDict( nextVal.get(), m_columnSources );
        return true;
    }

    const ColumnSourceMap & columnSources() const override
    {
        return m_columnSources;
    }

private:
    // Import a Python RecordBatchReader via Arrow C stream interface and wrap
    // it into a ColumnSource (schema from reader, readNext from reader).
    static std::shared_ptr<ColumnSource> importAsColumnSource( PyObject *pyReader )
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

        auto reader = result.MoveValueUnsafe();
        auto source = std::make_shared<ColumnSource>();
        source -> schema = reader -> schema();
        source -> readNext = [reader = std::move( reader )]( std::shared_ptr<::arrow::RecordBatch> * batch ) mutable
        {
            return reader -> ReadNext( batch );
        };
        return source;
    }

    static void importColumnSourceDict( PyObject *pyDict, ColumnSourceMap & out )
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
            out[ colName ] = importAsColumnSource( val );
        }
    }

    csp::python::PyObjectPtr m_factory;
    csp::python::PyObjectPtr m_iter;

    ColumnSourceMap m_columnSources;
};

// Native C++ parquet reader for regular and split-column parquet files.
// Produces ColumnSource objects backed by Arrow's async batch generator:
//   - Schema derived from file metadata via SchemaManifest (no I/O)
//   - readNext wraps the generator's Future-based iteration
//   - rows_to_readahead=1 pipelines decode of the next row group with engine processing
//   - pre_buffer=true (set in makeFileReader) pipelines I/O on Arrow's I/O pool
//
// Lifecycle safety:
//   - Destroying a ColumnSource while in-flight futures exist is safe: Arrow's generator
//     state is ref-counted (shared_ptr<MergedGenerator::State>), and pending pool
//     callbacks will complete independently then release their references.
//   - FileReader lifetime is managed by shared_ptr chains in both the generator's
//     internal captures and m_fileReaders/m_prevFileReaders.
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
        m_columnSources.clear();
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

    const ColumnSourceMap & columnSources() const override
    {
        return m_columnSources;
    }

private:
    // Count the number of leaf (primitive) columns an Arrow field contributes to parquet.
    static int countLeafColumns( const std::shared_ptr<::arrow::Field> & field )
    {
        auto type = field -> type();
        if( type -> id() == ::arrow::Type::STRUCT )
        {
            int count = 0;
            for( int i = 0; i < type -> num_fields(); ++i )
                count += countLeafColumns( type -> field( i ) );
            return count;
        }
        return 1;  // scalar or list types are one leaf in parquet
    }

    // Open a single parquet file, projecting only needed columns.
    bool openSingleFile( const std::string & path )
    {
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
                // Map from top-level Arrow schema fields to parquet leaf column indices.
                // Parquet stores nested struct fields as separate leaf columns.
                int leafIdx = 0;
                for( int i = 0; i < arrowSchema -> num_fields(); ++i )
                {
                    auto field = arrowSchema -> field( i );
                    auto fieldName = field -> name();
                    int numLeaves = countLeafColumns( field );

                    bool include = m_neededColumns.count( fieldName ) > 0;
                    if( !include )
                    {
                        // Include parent fields of needed nested columns (e.g. "struct.i" → "struct")
                        std::string prefix = fieldName + ".";
                        auto lb = m_neededColumns.lower_bound( prefix );
                        include = lb != m_neededColumns.end() && lb -> compare( 0, prefix.size(), prefix ) == 0;
                    }

                    if( include )
                    {
                        for( int leaf = 0; leaf < numLeaves; ++leaf )
                            colIndices.push_back( leafIdx + leaf );
                    }

                    leafIdx += numLeaves;
                }
            }
        }

        auto source = createBatchSource( fileReader, colIndices );
        if( !source )
            return false;

        // Use the first field name as the dict key (for single-file mode, the
        // manager iterates all fields from the schema anyway).
        m_columnSources[ source -> schema -> field( 0 ) -> name() ] = source;
        m_fileReaders.push_back( std::move( fileReader ) );
        return true;
    }

    // Open a split-column directory: each .parquet file is one column.
    bool openSplitDirectory( const std::string & dirPath )
    {
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

            auto source = createBatchSource( fileReader, {} );
            if( !source )
                continue;

            m_columnSources[colName] = source;
            m_fileReaders.push_back( std::move( fileReader ) );
        }

        return !m_columnSources.empty();
    }

    static std::shared_ptr<::parquet::arrow::FileReader> makeFileReader( const std::string & path )
    {
        auto fileResult = ::arrow::io::ReadableFile::Open( path );
        if( !fileResult.ok() )
            CSP_THROW( csp::ValueError, "Failed to open " << path << ": " << fileResult.status().ToString() );

        auto parquetReader = ::parquet::ParquetFileReader::Open( fileResult.ValueUnsafe() );

        ::parquet::ArrowReaderProperties arrowProps;
        arrowProps.set_use_threads( true );
        arrowProps.set_pre_buffer( true );

        std::unique_ptr<::parquet::arrow::FileReader> fileReader;
        auto status = ::parquet::arrow::FileReader::Make(
            ::arrow::default_memory_pool(), std::move( parquetReader ), arrowProps, &fileReader );
        if( !status.ok() )
            CSP_THROW( csp::ValueError, "Failed to create Arrow FileReader for " << path << ": " << status.ToString() );

        return fileReader;
    }

    // Compute the projected Arrow schema from parquet file metadata alone (no I/O).
    // Uses SchemaManifest to map leaf column indices → Arrow field indices.
    static std::shared_ptr<::arrow::Schema> getProjectedSchema(
        const std::shared_ptr<::parquet::arrow::FileReader> & fileReader,
        const std::vector<int> & resolvedCols )
    {
        auto metadata = fileReader -> parquet_reader() -> metadata();

        ::parquet::ArrowReaderProperties arrowProps;
        arrowProps.set_use_threads( true );
        arrowProps.set_pre_buffer( true );

        ::parquet::arrow::SchemaManifest manifest;
        auto status = ::parquet::arrow::SchemaManifest::Make(
            metadata -> schema(), metadata -> key_value_metadata(), arrowProps, &manifest );
        if( !status.ok() )
            CSP_THROW( csp::ValueError, "SchemaManifest::Make failed: " << status.ToString() );

        auto fieldIndicesResult = manifest.GetFieldIndices( resolvedCols );
        if( !fieldIndicesResult.ok() )
            CSP_THROW( csp::ValueError, "GetFieldIndices failed: " << fieldIndicesResult.status().ToString() );

        ::arrow::FieldVector fields;
        for( int idx : fieldIndicesResult.ValueUnsafe() )
            fields.push_back( manifest.schema_fields[idx].field );

        return ::arrow::schema( std::move( fields ), manifest.schema_metadata );
    }

    // Create a ColumnSource backed by Arrow's async record-batch generator.
    //   - Schema from metadata (getProjectedSchema)
    //   - readNext wraps the generator: calls operator() → blocks on Future → returns batch
    //   - rows_to_readahead=1: pipelines decode of next row group with engine processing
    //   - pre_buffer=true (set in makeFileReader): I/O on Arrow I/O pool (ARROW_IO_THREADS)
    //   - Decode uses CPU pool (OMP_NUM_THREADS) via non-blocking OptionalParallelForAsync
    static std::shared_ptr<ColumnSource> createBatchSource(
        const std::shared_ptr<::parquet::arrow::FileReader> & fileReader,
        const std::vector<int> & colIndices )
    {
        int numRG = fileReader -> num_row_groups();
        std::vector<int> rowGroups( numRG );
        std::iota( rowGroups.begin(), rowGroups.end(), 0 );

        // GetRecordBatchGenerator requires an explicit column list — an empty vector means
        // "no columns" (unlike the sync overload which defaults to all).
        std::vector<int> resolvedCols = colIndices;
        if( resolvedCols.empty() )
        {
            int numLeafCols = fileReader -> parquet_reader() -> metadata() -> num_columns();
            resolvedCols.resize( numLeafCols );
            std::iota( resolvedCols.begin(), resolvedCols.end(), 0 );
        }

        // Get projected schema from metadata (no I/O, no temporary reader).
        auto schema = getProjectedSchema( fileReader, resolvedCols );

        // Create the async generator.
        auto genResult = fileReader -> GetRecordBatchGenerator(
            fileReader, rowGroups, resolvedCols,
            ::arrow::internal::GetCpuThreadPool(), /*rows_to_readahead=*/1 );

        if( !genResult.ok() )
            CSP_THROW( csp::ValueError, "GetRecordBatchGenerator failed: " << genResult.status().ToString() );

        // Wrap generator in a readNext function.  The shared state (generator + fileReader)
        // is captured by value in the lambda — ref-counted, safe to copy.
        using Generator = std::function<::arrow::Future<std::shared_ptr<::arrow::RecordBatch>>()>;
        auto generator = std::make_shared<Generator>( std::move( genResult ).MoveValueUnsafe() );
        auto eof       = std::make_shared<bool>( false );

        auto source = std::make_shared<ColumnSource>();
        source -> schema = std::move( schema );
        source -> readNext = [generator, eof, fileReader]( std::shared_ptr<::arrow::RecordBatch> * batch ) -> ::arrow::Status
        {
            if( *eof )
            {
                *batch = nullptr;
                return ::arrow::Status::OK();
            }

            auto future = ( *generator )();
            auto result = future.MoveResult();

            if( !result.ok() )
            {
                *eof = true;
                return result.status();
            }

            *batch = result.MoveValueUnsafe();
            if( *batch == nullptr )
                *eof = true;

            return ::arrow::Status::OK();
        };

        return source;
    }

    csp::python::PyObjectPtr                                      m_filenameGen;
    bool                                                          m_splitColumns;
    bool                                                          m_allowMissingFiles;
    std::set<std::string>                                         m_neededColumns;
    std::vector<std::string>                                      m_filenames;
    size_t                                                        m_fileIdx = 0;
    ColumnSourceMap                                                m_columnSources;
    std::vector<std::shared_ptr<::parquet::arrow::FileReader>>    m_fileReaders;
    std::vector<std::shared_ptr<::parquet::arrow::FileReader>>    m_prevFileReaders;
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
