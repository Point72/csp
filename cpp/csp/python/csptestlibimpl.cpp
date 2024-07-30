#include <Python.h>
#include <csp/engine/CppNode.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyCppNode.h>
#include <csp/python/PyObjectPtr.h>

namespace csp::cppnodes
{

// Expose C++ nodes for testing in Python
// Keep nodes for a specific test to a single namespace under testing

namespace testing
{

namespace stop_start_test
{

using namespace csp::python;

void setStatus( const DialectGenericType & obj_, const std::string & name )
{
    PyObjectPtr obj = PyObjectPtr::own( toPython( obj_ ) );
    PyObjectPtr attr = PyObjectPtr::own( PyUnicode_FromString( name.c_str() ) );
    PyObject_SetAttr( obj.get(), attr.get(), Py_True );
}

DECLARE_CPPNODE( start_n1_set_value )
{
    INIT_CPPNODE( start_n1_set_value ) {}

    SCALAR_INPUT( DialectGenericType, obj_ );

    START()
    {
       setStatus( obj_, "n1_started" );
    }
    INVOKE() {}

    STOP()
    {
        setStatus( obj_, "n1_stopped" );
    }
};
EXPORT_CPPNODE( start_n1_set_value );

DECLARE_CPPNODE( start_n2_throw )
{
    INIT_CPPNODE( start_n2_throw ) {}

    SCALAR_INPUT( DialectGenericType, obj_ );

    START()
    {
        CSP_THROW( ValueError, "n2 start failed" );
    }
    INVOKE() {}

    STOP()
    {
        setStatus( obj_, "n2_stopped" );
    }
};
EXPORT_CPPNODE( start_n2_throw );

}

namespace interrupt_stop_test
{

using namespace csp::python;

void setStatus( const DialectGenericType & obj_, int64_t idx )
{
    PyObjectPtr obj = PyObjectPtr::own( toPython( obj_ ) );
    PyObjectPtr list = PyObjectPtr::own( PyObject_GetAttrString( obj.get(), "stopped" ) );
    PyList_SET_ITEM( list.get(), idx, Py_True );
}

DECLARE_CPPNODE( set_stop_index )
{
    INIT_CPPNODE( set_stop_index ) {}

    SCALAR_INPUT( DialectGenericType, obj_ );
    SCALAR_INPUT( int64_t, idx );

    START() {}
    INVOKE() {}

    STOP()
    {
       setStatus( obj_, idx );
    }
};
EXPORT_CPPNODE( set_stop_index );

}

}

}

// Test nodes
REGISTER_CPPNODE( csp::cppnodes::testing::stop_start_test, start_n1_set_value );
REGISTER_CPPNODE( csp::cppnodes::testing::stop_start_test, start_n2_throw );
REGISTER_CPPNODE( csp::cppnodes::testing::interrupt_stop_test, set_stop_index );

static PyModuleDef _csptestlibimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_csptestlibimpl",
    "_csptestlibimpl c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__csptestlibimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_csptestlibimpl_module);
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}
