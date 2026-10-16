#include <csp/adapters/csv/CsvInputAdapterManager.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/python/Conversions.h>
#include <csp/python/Exception.h>
#include <csp/python/InitHelper.h>
#include <csp/python/PyAdapterManagerWrapper.h>
#include <csp/python/PyEngine.h>
#include <csp/python/PyInputAdapterWrapper.h>
#include <csp/python/PyOutputAdapterWrapper.h>

using namespace csp::adapters::csv;

namespace csp::python {

// AdapterManager
csp::AdapterManager *create_csv_adapter_manager(PyEngine *engine,
                                                const Dictionary &properties) {
  return engine->engine()->createOwnedObject<CsvInputAdapterManager>(
      properties);
}

static InputAdapter *
create_csv_input_adapter(csp::AdapterManager *manager, PyEngine *pyengine,
                         PyObject *pyType, PushMode pushMode, PyObject *args) {
  auto &cspType = pyTypeAsCspType(pyType);

  PyObject *pyProperties;
  PyObject *type;

  auto *csvManager = dynamic_cast<CsvInputAdapterManager *>(manager);
  if (!csvManager)
    CSP_THROW(TypeError, "Expected CsvInputAdapterManager");

  if (!PyArg_ParseTuple(args, "O!O!", &PyType_Type, &type, &PyDict_Type,
                        &pyProperties))
    CSP_THROW(PythonPassthrough, "");

  return csvManager->getInputAdapter(
      cspType, fromPython<Dictionary>(pyProperties), pushMode);
}

REGISTER_ADAPTER_MANAGER(_csv_adapter_manager, create_csv_adapter_manager);
REGISTER_INPUT_ADAPTER(_csv_input_adapter, create_csv_input_adapter);

static PyModuleDef _csvadapterimpl_module = {PyModuleDef_HEAD_INIT,
                                             "_csvadapterimpl",
                                             "_csvadapterimpl c++ module",
                                             -1,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL,
                                             NULL};

PyMODINIT_FUNC PyInit__csvadapterimpl(void) {
  PyObject *m;

  m = PyModule_Create(&_csvadapterimpl_module);
  if (m == NULL)
    return NULL;

  if (!InitHelper::instance().execute(m))
    return NULL;

  return m;
}

} // namespace csp::python
