#ifndef _IN_CSP_PYTHON_PYADAPTERMANAGER_H
#define _IN_CSP_PYTHON_PYADAPTERMANAGER_H

#include <Python.h>

namespace csp { class AdapterManager; }

namespace csp::python
{

struct PyAdapterManager_PyObject
{
    PyObject_HEAD
    csp::AdapterManager * manager;

    static PyTypeObject PyType;
};

}

#endif
