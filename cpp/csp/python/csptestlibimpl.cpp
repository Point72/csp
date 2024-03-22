#include <Python.h>
#include <csp/engine/CppNode.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyCppNode.h>
#include <csp/python/PyObjectPtr.h>

// Test nodes
REGISTER_CPPNODE( csp::cppnodes::testing::stop_start_test, start_n1_set_value );
REGISTER_CPPNODE( csp::cppnodes::testing::stop_start_test, start_n2_throw );

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
