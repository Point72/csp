#ifndef _IN_CSP_PYTHON_PYADAPTERMANAGER_H
#define _IN_CSP_PYTHON_PYADAPTERMANAGER_H

#include <Python.h>
#include <csp/core/Platform.h>

namespace csp { class AdapterManager; }

namespace csp::python
{

struct CSPIMPL_EXPORT PyAdapterManager_PyObject
{
    PyObject_HEAD
    csp::AdapterManager * manager;

    static PyTypeObject PyType;
};

}

#endif
