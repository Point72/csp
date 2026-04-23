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
#include <arrow/io/memory.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <locale>
#include <codecvt>

using namespace csp::adapters::parquet;

//using namespace csp::cppnodes;
//namespace csp::adapters::parquet
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

// Generator that wraps a Python "stream factory" callable.
// The factory signature: factory(starttime, endtime, needed_columns) -> iterator of (reader, basket_dict)
// Each reader is a pyarrow.RecordBatchReader; basket_dict maps basket names to readers.
// Readers are imported via ArrowArrayStream for GIL-free batch pulling in C++.
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

        // Expect a tuple of (RecordBatchReader, dict)
        CSP_TRUE_OR_THROW( PyTuple_Check( nextVal.get() ) && PyTuple_GET_SIZE( nextVal.get() ) == 2,
                           csp::TypeError, "Stream factory expected to yield (reader, basket_dict) tuples" );

        PyObject *pyReader     = PyTuple_GET_ITEM( nextVal.get(), 0 );
        PyObject *pyBasketDict = PyTuple_GET_ITEM( nextVal.get(), 1 );

        // Import main reader via ArrowArrayStream
        m_mainReader = importRecordBatchReader( pyReader );

        // Import basket readers and read their single batch
        m_basketBatches.clear();
        if( PyDict_Check( pyBasketDict ) )
        {
            PyObject *key, *val;
            Py_ssize_t pos = 0;
            while( PyDict_Next( pyBasketDict, &pos, &key, &val ) )
            {
                const char *basketName = PyUnicode_AsUTF8( key );
                if( !basketName )
                    CSP_THROW( csp::python::PythonPassthrough, "" );

                auto basketReader = importRecordBatchReader( val );
                std::shared_ptr<::arrow::RecordBatch> batch;
                auto status = basketReader -> ReadNext( &batch );
                if( !status.ok() )
                    CSP_THROW( csp::ValueError, "Failed to read basket batch: " << status.ToString() );
                if( batch )
                    m_basketBatches[ basketName ] = batch;
            }
        }

        return true;
    }

    std::shared_ptr<::arrow::RecordBatchReader> mainReader() override
    {
        return m_mainReader;
    }

    const std::unordered_map<std::string, std::shared_ptr<::arrow::RecordBatch>> & basketBatches() const override
    {
        return m_basketBatches;
    }

private:
    static std::shared_ptr<::arrow::RecordBatchReader> importRecordBatchReader( PyObject *pyReader )
    {
        // Call __arrow_c_stream__() to export as ArrowArrayStream PyCapsule
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

    csp::python::PyObjectPtr m_factory;
    csp::python::PyObjectPtr m_iter;
    std::shared_ptr<::arrow::RecordBatchReader> m_mainReader;
    std::unordered_map<std::string, std::shared_ptr<::arrow::RecordBatch>> m_basketBatches;
};

}

namespace csp::python
{

//AdapterManager
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
//    PyObject *pyProperties;
//    PyObject *pyType;
//
//
//    if( !PyArg_ParseTuple( args, "O!O!",
//                           &PyType_Type, &pyType,
//                           &PyDict_Type, &pyProperties ))
//        CSP_THROW( PythonPassthrough, "" );
//
//    auto &cspType = pyTypeAsCspType( pyType );
//
//    return parquetManager->getOutputAdapter( cspType, fromPython<Dictionary>( pyProperties ));
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
        PyObject *pyFileGenerator = nullptr;

        if( !PyArg_ParseTuple( args, "O!O!O!",
                               &PyEngine::PyType, &pyEngine,
                               &PyDict_Type, &pyProperties,
                               &PyFunction_Type, &pyFileGenerator ) )
            CSP_THROW( PythonPassthrough, "" );

        auto streamSource = std::make_shared<PyRecordBatchStreamSource>( pyFileGenerator );
        auto *adapterMgr = create_parquet_input_adapter_manager_impl( pyEngine, fromPython<Dictionary>( pyProperties ),
                                                                      streamSource );
        auto res         = PyCapsule_New( adapterMgr, "adapterMgr", nullptr );
        return res;
    CSP_RETURN_NULL;
}

//AdapterManager
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
