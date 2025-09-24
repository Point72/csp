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
#include <csp/python/PyNodeWrapper.h>
#include <csp/python/NumpyConversions.h>
#include <arrow/c/abi.h>
#include <arrow/c/bridge.h>
#include <arrow/io/memory.h>
#include <arrow/table.h>
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
        if( nextValPtr.get() == nullptr )
        {
            return false;
        }

        if( !PyCapsule_IsValid( nextValPtr.get(), "arrow_array_stream" ) )
        {
            CSP_THROW( csp::TypeError, "Invalid arrow data, expected PyCapsule got " << Py_TYPE( nextValPtr.get() ) -> tp_name );
        }
        // Extract the record batch
        struct ArrowArrayStream * c_stream = reinterpret_cast<struct ArrowArrayStream*>( PyCapsule_GetPointer( nextValPtr.get(), "arrow_array_stream" ) );
        auto reader_result = arrow::ImportRecordBatchReader( c_stream );
        if( !reader_result.ok() )
            CSP_THROW( csp::ValueError, "Failed to load record batches through PyCapsule C Data interface: " << reader_result.status().ToString() );
        auto reader = std::move( reader_result.ValueUnsafe() );
        auto table_result = arrow::Table::FromRecordBatchReader( reader.get() );
        if( !table_result.ok() )
            CSP_THROW( csp::ValueError, "Failed to load table from record batches " << table_result.status().ToString() );
        value = std::move( table_result.ValueUnsafe() );
        return true;
    }
private:
    csp::python::PyObjectPtr m_wrappedGenerator;
    csp::python::PyObjectPtr m_iter;
};

template< typename CspCType>
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
        {
            CSP_THROW( csp::TypeError, "While writing to parquet expected numpy array type, got " << Py_TYPE( object ) -> tp_name );
        }

        PyArrayObject *arrayObject = ( PyArrayObject * ) ( object );
        char npy_type = PyArray_DESCR( arrayObject ) -> type;
        if( PyArray_DESCR( arrayObject ) -> kind != m_expectedArrayDesc -> kind )
        {
            CSP_THROW( csp::TypeError,
                       "Expected array of type " << csp::python::PyObjectPtr::own( PyObject_Repr( ( PyObject * ) m_expectedArrayDesc ) )
                                                 << " got "
                                                 << csp::python::PyObjectPtr::own( PyObject_Repr( ( PyObject * ) PyArray_DESCR( arrayObject ) ) ) );
        }

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
            {
                this->writeValue(static_cast<CspCType>(data[i]));
            }
        }
        else
        {
            for (decltype(arraySize) i = 0; i < arraySize; ++i)
            {
                this->writeValue(static_cast<CspCType>(*reinterpret_cast<NumpyCType*>(PyArray_GETPTR1(arrayObject, i))));
            }
        }
    }

    PyArray_Descr *m_expectedArrayDesc;
};

class NumpyUnicodeArrayWriter : public TypedDialectGenericListWriterInterface<std::string>
{
public:
    NumpyUnicodeArrayWriter()
    {
    }

    void writeItems( const csp::DialectGenericType &listObject ) override
    {
        PyObject *object = csp::python::toPythonBorrowed( listObject );

        if( !PyArray_Check( object ) )
        {
            CSP_THROW( csp::TypeError, "While writing to parquet expected numpy array type, got " << Py_TYPE( object ) -> tp_name );
        }
        PyArrayObject *arrayObject = ( PyArrayObject * ) ( object );

        if( PyArray_DESCR( arrayObject ) -> type_num != NPY_UNICODE )
        {
            CSP_THROW( csp::TypeError,
                       "Expected array of type " << csp::python::PyObjectPtr::own( PyObject_Repr( ( PyObject * ) m_expectedArrayDesc ) )
                                                 << " got "
                                                 << csp::python::PyObjectPtr::own(
                                                         PyObject_Repr( ( PyObject * ) PyArray_DESCR( arrayObject ) ) ) );
        }

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

private:
    PyArray_Descr *m_expectedArrayDesc;
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
                    {
                        return std::make_shared<NumpyUnicodeArrayWriter>();
                    }
                    else
                    {
                        return std::make_shared<NumpyArrayWriterImpl<CValueType>>(numpy_dtype);
                    }
                }
        );
    }
    catch( csp::TypeError &e )
    {
        CSP_THROW( csp::TypeError, "Unsupported array value type when writing to parquet:" << type -> type().asString() );
    }
}


template< typename V >
class NumpyArrayReaderImpl final : public TypedDialectGenericListReaderInterface<V>
{
public:
    NumpyArrayReaderImpl( PyArray_Descr *expectedArrayDesc )
    : m_expectedArrayDesc( expectedArrayDesc )
    {
    }
    virtual csp::DialectGenericType create(uint32_t size) override
    {
        npy_intp iSize = size;

        Py_INCREF(m_expectedArrayDesc);
        PyObject* arr = PyArray_SimpleNewFromDescr( 1, &iSize, m_expectedArrayDesc );
        // Since arr already has reference count
        csp::python::PyObjectPtr objectPtr{csp::python::PyObjectPtr::own(arr)};

        // We need to make sure that's the case, since we are going to return pointer to raw buffer
        CSP_ASSERT(PyArray_ISCARRAY( reinterpret_cast<PyArrayObject *>(arr)));

        csp::DialectGenericType res{csp::python::fromPython<csp::DialectGenericType>(arr)};
        return res;
    }

    csp::DialectGenericType create( uint32_t size, uint32_t maxElementSize ) override
    {
        CSP_NOT_IMPLEMENTED;
    }

    virtual V *getRawDataBuffer( const csp::DialectGenericType &list ) const override
    {
        auto arrayObject = reinterpret_cast<PyArrayObject *>(csp::python::toPythonBorrowed(list));
        return reinterpret_cast<V *>(PyArray_DATA( arrayObject ));
    }

    virtual void setValue(const csp::DialectGenericType& list, int index, const V& value) override
    {
        getRawDataBuffer(list)[index] = value;
    }

private:
    PyArray_Descr *m_expectedArrayDesc;
};

class NumpyUnicodeReaderImpl final : public TypedDialectGenericListReaderInterface<std::string>
{
public:
    NumpyUnicodeReaderImpl()
    {
    }

    virtual csp::DialectGenericType create( uint32_t size ) override
    {
        CSP_NOT_IMPLEMENTED;
    }

    csp::DialectGenericType create( uint32_t size, uint32_t maxElementSize ) override
    {
        npy_intp iSize = size;

        PyArray_Descr *typ;
        PyObject      *type_string_descr = csp::python::toPython( std::string( "U" ) + std::to_string( maxElementSize ) );
        PyArray_DescrConverter( type_string_descr, &typ );
        Py_DECREF( type_string_descr );

        PyObject *arr = PyArray_SimpleNewFromDescr( 1, &iSize, typ );

        // Since arr already has reference count
        csp::python::PyObjectPtr objectPtr{ csp::python::PyObjectPtr::own( arr ) };

        csp::DialectGenericType res{ csp::python::fromPython<csp::DialectGenericType>( arr ) };
        return res;
    }

    std::string *getRawDataBuffer( const csp::DialectGenericType &list ) const override
    {
        return nullptr;
    }

    void setValue( const csp::DialectGenericType &list, int index, const std::string &value ) override
    {
        auto arrayObject = reinterpret_cast<PyArrayObject *>(csp::python::toPythonBorrowed( list ));
        std::wstring_convert<std::codecvt_utf8<char32_t>,char32_t> converter;
        auto elementSize = PyDataType_ELSIZE( PyArray_DESCR( arrayObject ) );
        auto wideValue = converter.from_bytes( value );
        auto nElementsToCopy = std::min( int(elementSize / sizeof(char32_t)), int( wideValue.size() + 1 ) );
        std::copy_n( wideValue.c_str(), nElementsToCopy, reinterpret_cast<char32_t*>(PyArray_GETPTR1( arrayObject, index )) );
    }
};


inline DialectGenericListReaderInterface::Ptr create_numpy_array_reader_impl( const csp::CspTypePtr &type )

{
    try
    {
        return csp::PartialSwitchCspType<csp::CspType::Type::DOUBLE, csp::CspType::Type::INT64,
                csp::CspType::Type::BOOL, csp::CspType::Type::STRING>::invoke( type.get(),
                                                                               []( auto tag ) -> DialectGenericListReaderInterface::Ptr
                                                                               {
                                                                                   using TagType = decltype(tag);
                                                                                   using CValueType = typename TagType::type;
                                                                                   auto numpy_dtype = PyArray_DescrFromType(
                                                                                           csp::python::NPY_TYPE<CValueType>::value );

                                                                                   if( numpy_dtype -> type_num == NPY_UNICODE )
                                                                                   {
                                                                                       return std::make_shared<NumpyUnicodeReaderImpl>();
                                                                                   }
                                                                                   else
                                                                                   {
                                                                                       return std::make_shared<NumpyArrayReaderImpl<CValueType>>(
                                                                                               numpy_dtype );
                                                                                   }
                                                                               }
        );
    }
    catch( csp::TypeError &e )
    {
        CSP_THROW( csp::TypeError, "Unsupported array value type when reading from parquet:" << type -> type().asString() );
    }
}

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
                                                  create_numpy_array_reader_impl( valueType ) );
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
