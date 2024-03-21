#include <csp/core/Exports.h>
#include <csp/python/InitHelper.h>
#include <frameobject.h>
#include <traceback.h>
#include "Python.h"
#include "datetime.h"

namespace csp::python
{

static PyModuleDef _csptypesimpl_module = {
    PyModuleDef_HEAD_INIT,
    "_csptypesimpl",
    "_csptypesimpl c++ module",
    -1,
    NULL, NULL, NULL, NULL, NULL
};

extern "C" CSP_PYTHON_TYPES_EXPORT PyObject* PyInit__csptypesimpl(void)
{
    PyDateTime_IMPORT;
    PyObject* m;

    m = PyModule_Create( &_csptypesimpl_module);
    if( m == NULL )
        return NULL;

    if( !InitHelper::instance().execute( m ) )
        return NULL;

    return m;
}

}
