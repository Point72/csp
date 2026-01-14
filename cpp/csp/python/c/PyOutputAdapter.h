#ifndef _IN_CSP_PYTHON_C_PYOUTPUTADAPTER_H
#define _IN_CSP_PYTHON_C_PYOUTPUTADAPTER_H

#include "Python.h"

// Create a special python capsule type to indicate to the C++ code that this is an output adapter
// defined in an external language and communicating via the C ABI interface.

const char * const CSP_PYTHON_C_OUTPUT_ADAPTER_CAPSULE_NAME = "csp.python.c.OutputAdapterCapsule";

PyObject * createCOutputAdapterCapsule( OutputAdapter * c_adapter_ptr ) {
    return PyCapsule_New( c_adapter_ptr, CSP_PYTHON_C_OUTPUT_ADAPTER_CAPSULE_NAME, NULL );
}

#endif
