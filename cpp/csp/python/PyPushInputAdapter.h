#ifndef _IN_CSP_PYTHON_PYPUSHINPUTADAPTER_H
#define _IN_CSP_PYTHON_PYPUSHINPUTADAPTER_H

#include <csp/core/Platform.h>
#include <csp/engine/PushInputAdapter.h>

namespace csp::python
{

//PushBatch
struct CSP_PUBLIC PyPushBatch
{
    PyObject_HEAD
    PushBatch batch;

    static PyTypeObject PyType;
};

}

#endif
