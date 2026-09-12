#include <csp/python/Conversions.h>
#include <csp/python/CspTypeFactory.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyCspEnum.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::python
{

DialectCspEnumMeta::DialectCspEnumMeta( PyTypeObjectPtr pyType, const std::string & name,
                                        const CspEnumMeta::ValueDef & def ) :
    CspEnumMeta( name, def ),
    m_pyType( pyType )
{
    //pre-create instances
    m_enumsByName  = PyObjectPtr::own( PyDict_New() );

    PyObjectPtr iter = PyObjectPtr::check( PyObject_GetIter( ( PyObject * ) pyType.get() ) );
    PyObjectPtr member;
    while( ( member = PyObjectPtr::own( PyIter_Next( iter.get() ) ) ) )
    {
        PyObjectPtr name = PyObjectPtr::check( PyObject_GetAttrString( member.get(), "name" ) );

        if( !PyLong_Check( member.get() ) )
            CSP_THROW( TypeError, "enum key " << name << " expected an integer got " << member );
        
        int64_t value = fromPython<int64_t>( member.get() );
        m_enumsByCValue[ value ] = member;

        if( PyDict_SetItem( m_enumsByName.get(), name.get(), member.get() ) < 0 )
            CSP_THROW( PythonPassthrough, "" );
    }
}

}
