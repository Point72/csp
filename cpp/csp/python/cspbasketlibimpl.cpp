#include <csp/engine/CppNode.h>
#include <csp/python/PyCppNode.h>

REGISTER_CPPNODE( csp::cppnodes, _sync_list );
REGISTER_CPPNODE( csp::cppnodes, _sample_list );

static PyModuleDef _cspbasketlibimpl_module = {
        PyModuleDef_HEAD_INIT,
        "_cspbasketlibimpl",
        "_cspbasketlibimpl c++ module",
        -1,
        NULL, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cspbasketlibimpl(void)
{
    PyObject* m;

    m = PyModule_Create( &_cspbasketlibimpl_module);
    if( m == NULL )
        return NULL;

    if( !csp::python::InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}