#ifndef _IN_CSP_PYTHON_PYOBJECTPTR_H
#define _IN_CSP_PYTHON_PYOBJECTPTR_H

#include <csp/python/Exception.h>

namespace csp::python
{

struct PyObjectOwn {};

template<typename PYOBJECT_T>
class PyPtr
{
public:
    PyPtr() : m_obj( nullptr ) {}
    ~PyPtr() { Py_XDECREF( m_obj ); }

    PyPtr( const PyPtr & rhs ) : PyPtr( rhs.m_obj ) {}
    PyPtr( PyPtr && rhs ) : m_obj( rhs.m_obj ) 
    {
        rhs.m_obj = nullptr;
    }

    PyPtr & operator=( const PyPtr & rhs )
    {
        Py_XDECREF( m_obj );
        m_obj = rhs.m_obj;
        Py_XINCREF( rhs.m_obj );
        return *this;
    }

    //avoid incref on move
    PyPtr & operator=( PyPtr && rhs )
    {
        Py_XDECREF( m_obj );
        m_obj = rhs.m_obj;
        rhs.m_obj = nullptr;
        return *this;
    }

    bool operator==( const PyPtr & rhs ) const { return generic_compare( rhs, Py_EQ ); }

    bool operator<( const PyPtr & rhs ) const { return generic_compare( rhs, Py_LT ); }

    bool operator>( const PyPtr & rhs ) const { return generic_compare( rhs, Py_GT ); }

    operator bool() const { return m_obj != nullptr; }

    PYOBJECT_T * get()        const { return m_obj; }
    PYOBJECT_T * ptr()        const { return m_obj; }
    PYOBJECT_T * operator->() const { return m_obj; }

    PYOBJECT_T * release()
    {
        PYOBJECT_T * rv = m_obj;
        m_obj = nullptr;
        return rv;
    }

    void reset()
    {
        Py_XDECREF( m_obj );
        m_obj = nullptr;
    }

    PyPtr incref() const                { return PyPtr( m_obj ); }

    size_t hash() const
    {
        if( !m_obj )
            return 0;

        Py_hash_t hash = PyObject_Hash( m_obj );
        if( hash == -1 )
            CSP_THROW( PythonPassthrough, "" );
        return hash;
    }

    static PyPtr own( PYOBJECT_T * o )    { return PyPtr( ( PyObjectOwn * ) o ); }
    static PyPtr incref( PYOBJECT_T * o ) { return PyPtr( o ); }

    //expects a new reference but if null throws passthrough exception, otherwise takes ownership
    static PyPtr check( PYOBJECT_T * o )  
    {
        if( !o )
            CSP_THROW( PythonPassthrough, "" );

        return PyPtr( ( PyObjectOwn * ) o ); 
    }

private:
    PyPtr( PYOBJECT_T * o )  { Py_XINCREF( o ); m_obj = o; }
    PyPtr( PyObjectOwn * o ) { m_obj = ( PYOBJECT_T * ) o; }

    PYOBJECT_T * m_obj;

    bool generic_compare( const PyPtr & rhs, int opid ) const
    {
        if( !m_obj || !rhs.m_obj )
            CSP_THROW( PythonPassthrough, "" );
        
        int rv = PyObject_RichCompareBool( m_obj, rhs.m_obj, opid );
        if( rv == -1 )
            CSP_THROW( PythonPassthrough, "" );
        return rv;
    }
};

using PyObjectPtr     = PyPtr<PyObject>;
using PyTypeObjectPtr = PyPtr<PyTypeObject>;

template<typename PYOBJECT_T>
inline std::ostream & operator<<( std::ostream & o, const PyPtr<PYOBJECT_T> & obj )
{
    o << PyUnicode_AsUTF8( PyPtr<PYOBJECT_T>::own( PyObject_Str( obj.ptr() ) ).ptr() );
    return o;
}

}

namespace std
{

template<>
struct hash<csp::python::PyObjectPtr>
{
    size_t operator()( const csp::python::PyObjectPtr & p ) const
    {
        return p.hash();
    }
};

}

#endif
