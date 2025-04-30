#include <numpy/ndarrayobject.h>
#include <csp/adapters/parquet/ParquetInputAdapterManager.h>
#include <csp/adapters/parquet/ParquetOutputAdapterManager.h>
#include <csp/adapters/parquet/ParquetDictBasketOutputWriter.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/Generator.h>
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
#include <csp/python/PyDialectGenericListsInterface.h>
#include <csp/python/NumpyConversions.h>
#include <arrow/io/memory.h>
#include <arrow/ipc/reader.h>
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
        if( unlikely( filename_provider.ticked() ) )
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

class FileNameGenerator : public csp::Generator<std::string, csp::DateTime, csp::DateTime>
{
public:

    FileNameGenerator( PyObject *wrappedGenerator )
            : m_wrappedGenerator( csp::python::PyObjectPtr::incref( wrappedGenerator ) )
    {
    }

    void init( csp::DateTime start, csp::DateTime end ) override
    {
        PyObject *tp = PyTuple_New( 2 );
        if( !tp )
            CSP_THROW( csp::python::PythonPassthrough, "" );

        PyTuple_SET_ITEM( tp, 0, csp::python::toPython( start ) );
        PyTuple_SET_ITEM( tp, 1, csp::python::toPython( end ) );
        m_iter = csp::python::PyObjectPtr::check( PyObject_Call( m_wrappedGenerator.ptr(), tp, nullptr ) );
        CSP_TRUE_OR_THROW( PyIter_Check( m_iter.get() ), csp::TypeError,
                           "Parquet file generator expected to return iterator" );
    }

    virtual bool next( std::string &value ) override
    {
        if( m_iter.ptr() == nullptr )
        {
            return false;
        }
        auto nextVal = csp::python::PyObjectPtr::own( PyIter_Next( m_iter.ptr() ) );
        if( PyErr_Occurred() )
        {
            CSP_THROW( csp::python::PythonPassthrough, "" );
        }
        if( nextVal.get() == nullptr )
        {
            return false;
        }
        value = csp::python::fromPython<std::string>( nextVal.get() );
        return true;
    }

private:
    csp::python::PyObjectPtr m_wrappedGenerator;
    csp::python::PyObjectPtr m_iter;
};


class ArrowTableGenerator : public csp::Generator<std::shared_ptr<arrow::Table>, csp::DateTime, csp::DateTime>
{
public:
    ArrowTableGenerator( PyObject *wrappedGenerator )
            : m_wrappedGenerator( csp::python::PyObjectPtr::incref( wrappedGenerator ) )
    {
    }

    void init( csp::DateTime start, csp::DateTime end ) override
    {
        PyObject *tp = PyTuple_New( 2 );
        if( !tp )
            CSP_THROW( csp::python::PythonPassthrough, "" );

        PyTuple_SET_ITEM( tp, 0, csp::python::toPython( start ) );
        PyTuple_SET_ITEM( tp, 1, csp::python::toPython( end ) );
        m_iter = csp::python::PyObjectPtr::check( PyObject_Call( m_wrappedGenerator.ptr(), tp, nullptr ) );
        CSP_TRUE_OR_THROW( PyIter_Check( m_iter.ptr() ), csp::TypeError,
                           "Parquet file generator expected to return iterator" );
    }

    virtual bool next( std::shared_ptr<arrow::Table> &value ) override
    {
        if( m_iter.ptr() == nullptr )
        {
            return false;
        }
        auto nextVal    = PyIter_Next( m_iter.ptr() );
        auto nextValPtr = csp::python::PyObjectPtr::own( nextVal );
        if( PyErr_Occurred() )
        {
            CSP_THROW( csp::python::PythonPassthrough, "" );
        }
        if( nextVal == nullptr )
        {
            return false;
        }

        if(!PyBytes_Check( nextVal ))
        {
            CSP_THROW( csp::TypeError, "Invalid arrow buffer type, expected bytes got " << Py_TYPE( nextVal ) -> tp_name );
        }
        const char * data = PyBytes_AsString( nextVal );
        if( !data )
            CSP_THROW( csp::python::PythonPassthrough, "" );
        auto size = PyBytes_Size(nextVal);
        m_data = csp::python::PyObjectPtr::incref(nextVal);
        std::shared_ptr<arrow::io::BufferReader> bufferReader = std::make_shared<arrow::io::BufferReader>(
                reinterpret_cast<const uint8_t *>(data), size );
        std::shared_ptr<arrow::ipc::RecordBatchStreamReader> reader = arrow::ipc::RecordBatchStreamReader::Open(bufferReader.get()).ValueOrDie();
        auto result = reader->ToTable();
        if (!(result.ok()))
            CSP_THROW(csp::RuntimeException, "Failed read arrow table from buffer");
        value = std::move(result.ValueUnsafe());
        return true;
    }
private:
    csp::python::PyObjectPtr m_wrappedGenerator;
    csp::python::PyObjectPtr m_iter;
    // We need to keep the last buffer in memory since arrow doesn't copy it but can refer to strings in it
    csp::python::PyObjectPtr m_data;
};

}

namespace csp::python
{

//AdapterManager
csp::AdapterManager *create_parquet_input_adapter_manager_impl( PyEngine *engine, const Dictionary &properties,
                                                                FileNameGenerator::Ptr fileNameGenerator,
                                                                ArrowTableGenerator::Ptr arrowTableGenerator )
{
    auto res = engine -> engine() -> createOwnedObject<ParquetInputAdapterManager>( properties, fileNameGenerator, arrowTableGenerator );
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
        auto &&valueType = pyTypeAsCspType( toPythonBorrowed( propertiesDict.get<DialectGenericType>( "array_value_type" ) ) );
        return parquetManager -> getInputAdapter( valueType, propertiesDict, pushMode,
                                                  csp::python::create_numpy_array_reader_impl( valueType ) );
    }
    else
    {
        return parquetManager -> getInputAdapter( cspType, propertiesDict, pushMode );
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
        return parquetManager -> getListOutputAdapter( valueType, propertiesDict, csp::python::create_numpy_array_writer_impl( valueType ) );
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

        std::shared_ptr<FileNameGenerator>   fileNameGenerator;
        std::shared_ptr<ArrowTableGenerator> arrowTableGenerator;

        auto dictionary = fromPython<Dictionary>( pyProperties );
        if( dictionary.get<bool>( "read_from_memory_tables" ) )
        {
            arrowTableGenerator = std::make_shared<ArrowTableGenerator>( pyFileGenerator );
        }
        else
        {
            fileNameGenerator = std::make_shared<FileNameGenerator>( pyFileGenerator );
        }
        auto *adapterMgr = create_parquet_input_adapter_manager_impl( pyEngine, fromPython<Dictionary>( pyProperties ),
                                                                      fileNameGenerator, arrowTableGenerator );
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

    return m;
}

}
