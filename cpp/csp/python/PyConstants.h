#ifndef _IN_CSP_PYTHON_PYCONSTANTS_H
#define _IN_CSP_PYTHON_PYCONSTANTS_H

#include <object.h>

namespace csp::python::constants
{
//These return borrowed refs
PyObject * UNSET();
PyObject * REMOVE_DYNAMIC_KEY();
//not really a constant, but convenient place to put this
PyObject * EDGE_TYPE();

}

#endif

