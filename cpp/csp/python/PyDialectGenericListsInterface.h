#ifndef _IN_CSP_PYTHON_DIALECT_GENERIC_INTERFACE_H
#define _IN_CSP_PYTHON_DIALECT_GENERIC_INTERFACE_H

#include <numpy/ndarrayobject.h>
#include <csp/adapters/parquet/DialectGenericListReaderInterface.h>
#include <csp/adapters/parquet/DialectGenericListWriterInterface.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/engine/CspType.h>
#include <csp/python/NumpyConversions.h>
#include <locale>
#include <codecvt>

namespace csp::python
{

template< typename CspCType>
class NumpyArrayWriterImpl : public csp::adapters::parquet::TypedDialectGenericListWriterInterface<CspCType>
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

class NumpyUnicodeArrayWriter : public csp::adapters::parquet::TypedDialectGenericListWriterInterface<std::string>
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

        auto elementSize = PyArray_DESCR( arrayObject ) -> elsize;
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

inline csp::adapters::parquet::DialectGenericListWriterInterface::Ptr create_numpy_array_writer_impl( const csp::CspTypePtr &type )
{
    try
    {
        return csp::PartialSwitchCspType<csp::CspType::Type::DOUBLE, csp::CspType::Type::INT64,
                csp::CspType::Type::BOOL, csp::CspType::Type::STRING>::invoke(
                type.get(),
                []( auto tag ) -> csp::adapters::parquet::DialectGenericListWriterInterface::Ptr
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
class NumpyArrayReaderImpl final : public csp::adapters::parquet::TypedDialectGenericListReaderInterface<V>
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

class NumpyUnicodeReaderImpl final : public csp::adapters::parquet::TypedDialectGenericListReaderInterface<std::string>
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
        auto elementSize = PyArray_DESCR( arrayObject ) -> elsize;
        auto wideValue = converter.from_bytes( value );
        auto nElementsToCopy = std::min( int(elementSize / sizeof(char32_t)), int( wideValue.size() + 1 ) );
        std::copy_n( wideValue.c_str(), nElementsToCopy, reinterpret_cast<char32_t*>(PyArray_GETPTR1( arrayObject, index )) );
    }
};


inline csp::adapters::parquet::DialectGenericListReaderInterface::Ptr create_numpy_array_reader_impl( const csp::CspTypePtr &type )

{
    try
    {
        return csp::PartialSwitchCspType<csp::CspType::Type::DOUBLE, csp::CspType::Type::INT64,
                csp::CspType::Type::BOOL, csp::CspType::Type::STRING>::invoke( type.get(),
                                                                               []( auto tag ) -> csp::adapters::parquet::DialectGenericListReaderInterface::Ptr
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

#endif
