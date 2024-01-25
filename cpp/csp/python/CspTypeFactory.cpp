#include <csp/python/Conversions.h>
#include <csp/python/CspTypeFactory.h>
#include <csp/python/PyStruct.h>
#include <datetime.h>

namespace csp::python
{

CspTypeFactory & CspTypeFactory::instance()
{
    static CspTypeFactory s_instance;
    return s_instance;
}

CspTypePtr & CspTypeFactory::typeFromPyType( PyObject * pyTypeObj )
{
    // List objects shouldn't be cached since they are temporary objects
    if( PyList_Check( pyTypeObj ) )
    {
        if( PyList_GET_SIZE( ( PyObject * ) pyTypeObj ) != 1 )
            CSP_THROW( TypeError, "Expected list types to be single element of sub-type" );

        PyObject *pySubType = PyList_GET_ITEM( pyTypeObj, 0 );
        if( !PyType_Check( pySubType ) )
            CSP_THROW( TypeError, "nested typed lists are not supported" );
        CspTypePtr elemType = typeFromPyType( pySubType );
        return CspArrayType::create( elemType );
    }

    PyTypeObject *pyType = (PyTypeObject*) pyTypeObj;
    auto rv = m_cache.emplace( pyType, nullptr );
    if( rv.second )
    {
        if( pyType == &PyFloat_Type )
            rv.first -> second = csp::CspType::DOUBLE();
        else if( pyType == &PyLong_Type )
            rv.first -> second = csp::CspType::INT64();
        else if( pyType == &PyBool_Type )
            rv.first -> second = csp::CspType::BOOL();
        else if( pyType == &PyUnicode_Type )
            rv.first -> second = csp::CspType::STRING();
        else if( pyType == &PyBytes_Type )
            rv.first -> second = csp::CspType::BYTES();
        else if( PyType_IsSubtype( pyType, &PyStruct::PyType ) )
        {
            auto meta = ( ( PyStructMeta * ) pyType ) -> structMeta;
            rv.first -> second = std::make_shared<csp::CspStructType>( meta );
        }
        else if( PyType_IsSubtype( pyType, &PyCspEnum::PyType ) )
        {
            auto meta = ( ( PyCspEnumMeta * ) pyType ) -> enumMeta;
            rv.first -> second = std::make_shared<csp::CspEnumType>( meta );
        }
        else if( pyType == PyDateTimeAPI -> DateTimeType )
            rv.first -> second = csp::CspType::DATETIME();
        else if( pyType == PyDateTimeAPI -> DeltaType )
            rv.first -> second = csp::CspType::TIMEDELTA();
        else if( pyType == PyDateTimeAPI -> DateType )
            rv.first -> second = csp::CspType::DATE();
        else if( pyType == PyDateTimeAPI -> TimeType )
            rv.first -> second = csp::CspType::TIME();
        else
        {
            if( !PyType_Check( pyType ) )
                CSP_THROW( TypeError, "expected python type for CspType got " << PyObjectPtr::incref( ( PyObject * ) pyType ) );
            rv.first -> second = csp::CspType::DIALECT_GENERIC();
        }
    }

    return rv.first -> second;
}
 
void CspTypeFactory::removeCachedType( PyTypeObject * pyType )
{
    m_cache.erase( pyType );
}

}
