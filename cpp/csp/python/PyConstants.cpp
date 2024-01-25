#include <Python.h>
#include <pythonrun.h>
#include <csp/python/Conversions.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyConstants.h>

static PyObject * s_UNSET;
static PyObject * s_REMOVE_DYNAMIC_KEY;
static PyObject * s_EDGE_TYPE;

namespace csp::python::constants
{
    PyObject * UNSET() 
    {
        if( unlikely( s_UNSET == nullptr ) )
        {
            PyObject * wiring = PyImport_AddModule("csp.impl.constants");
            PyObject * dict = PyModule_GetDict(wiring);
            s_UNSET = PyDict_GetItemString(dict, "UNSET");
            Py_INCREF( s_UNSET );
        }
        return s_UNSET;
    }

    PyObject * REMOVE_DYNAMIC_KEY() 
    {
        if( unlikely( s_REMOVE_DYNAMIC_KEY == nullptr ) )
        {
            PyObject * wiring = PyImport_AddModule("csp.impl.constants");
            PyObject * dict = PyModule_GetDict(wiring);
            s_REMOVE_DYNAMIC_KEY = PyDict_GetItemString(dict, "REMOVE_DYNAMIC_KEY");
            Py_INCREF( s_REMOVE_DYNAMIC_KEY );
        }
        return s_REMOVE_DYNAMIC_KEY;
    }

    PyObject * EDGE_TYPE()
    {
        if( unlikely( s_EDGE_TYPE == nullptr ) )
        {
            PyObject * mod = PyImport_AddModule("csp.impl.wiring.edge");
            PyObject * dict = PyModule_GetDict( mod );
            s_EDGE_TYPE = PyDict_GetItemString( dict, "Edge" );
            Py_INCREF( s_EDGE_TYPE );
        }

        return s_EDGE_TYPE;
    }
}
