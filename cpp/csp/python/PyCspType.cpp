#include <csp/python/PyCspType.h>
#include <csp/python/PyStruct.h>
#include <csp/python/Conversions.h>
#include <Python.h>

static_assert( sizeof( csp::DialectGenericType ) == sizeof( csp::python::PyObjectPtr ) );
static_assert( alignof( csp::DialectGenericType ) == alignof( csp::python::PyObjectPtr ) );

namespace csp
{
DialectGenericType::DialectGenericType()
{
    new( this ) csp::python::PyObjectPtr();
}

DialectGenericType::~DialectGenericType()
{
    using T = csp::python::PyObjectPtr;
    reinterpret_cast<T *>(this) -> ~T();
}

DialectGenericType::DialectGenericType( const DialectGenericType &rhs )
{
    new( this ) csp::python::PyObjectPtr( reinterpret_cast<const csp::python::PyObjectPtr &>(rhs) );
}

DialectGenericType::DialectGenericType( DialectGenericType &&rhs )
{
    new( this ) csp::python::PyObjectPtr( reinterpret_cast<csp::python::PyObjectPtr &&>(rhs) );
}

DialectGenericType DialectGenericType::deepcopy() const
{
    static PyObject * pyDeepcopy = PyObject_GetAttrString( PyImport_ImportModule( "copy" ), "deepcopy" );
    PyObject * pyVal = PyObject_CallFunction( pyDeepcopy, "(O)", python::toPythonBorrowed( *this ) );
    return DialectGenericType( reinterpret_cast<DialectGenericType &&>( std::move( csp::python::PyObjectPtr::check( pyVal ) ) ) );
}

DialectGenericType &DialectGenericType::operator=( const DialectGenericType &rhs )
{
    *reinterpret_cast<csp::python::PyObjectPtr *>(this) = reinterpret_cast<const csp::python::PyObjectPtr &>(rhs);
    return *this;
}

DialectGenericType &DialectGenericType::operator=( DialectGenericType &&rhs )
{
    *reinterpret_cast<csp::python::PyObjectPtr *>(this) = std::move( reinterpret_cast<csp::python::PyObjectPtr &&>(rhs) );
    return *this;
}


bool DialectGenericType::operator==( const DialectGenericType &rhs ) const
{
    return *reinterpret_cast<const csp::python::PyObjectPtr *>(this) == reinterpret_cast<const csp::python::PyObjectPtr &>(rhs);
}

size_t DialectGenericType::hash() const
{
    return reinterpret_cast<const csp::python::PyObjectPtr *>(this) -> hash();
}

std::ostream & operator<<( std::ostream & o, const DialectGenericType & obj )
{
    o << reinterpret_cast<const python::PyObjectPtr &>( obj );
    return o;
}

}
