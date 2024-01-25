#ifndef _IN_CSP_PYTHON_PYPUSHINPUTADAPTER_H
#define _IN_CSP_PYTHON_PYPUSHINPUTADAPTER_H

#include <csp/engine/PushInputAdapter.h>

namespace csp::python
{

//PushBatch
struct PyPushBatch
{
    PyObject_HEAD
    PushBatch batch;

    static PyTypeObject PyType;
};

}

#endif
