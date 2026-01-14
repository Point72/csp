#include <stddef.h>
#include "Python.h"

#include <csp/engine/c/OutputAdapter.h>
#include <csp/python/c/PyOutputAdapter.h>

PyObject* create_input_adapter_py(PyObject*, PyObject*) {
    // TODO
    return Py_None;
}
PyObject* create_output_adapter_py(PyObject*, PyObject*) {
    // TODO
    uint64_t x = 42;
    void* ptr = (void*) x;
    return createCOutputAdapterCapsule( (OutputAdapter*) x );
}

static PyMethodDef exampleadapter_methods[] = {
  {"_example_input_adapter", (PyCFunction)create_input_adapter_py, METH_VARARGS},
  {"_example_output_adapter", (PyCFunction)create_output_adapter_py, METH_VARARGS},
  {NULL, NULL, 0, NULL}
};

static PyModuleDef exampleadapter_module = {
  PyModuleDef_HEAD_INIT,
  "exampleadapter",
  "exampleadapter module",
  -1,
  exampleadapter_methods
};
  
  
PyMODINIT_FUNC PyInit__exampleadapterimpl(void) {
  Py_Initialize();
  return PyModule_Create(&exampleadapter_module);
}