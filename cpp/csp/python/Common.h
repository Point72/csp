#ifndef _IN_CSP_PYTHON_COMMON_H
#define _IN_CSP_PYTHON_COMMON_H

#include <Python.h>

#define IS_PRE_PYTHON_3_11 (PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 11) )

#define INIT_PYDATETIME if( !PyDateTimeAPI ) { PyDateTime_IMPORT; }

// NumPy 2.0 Migration
#include <numpy/numpyconfig.h>

#if NPY_ABI_VERSION >= 0x02000000
// Define helper for anything that can't
// be handled by the below helper macros
#define CSP_NUMPY_2

#else

// Numpy 2.0 helpers
#define PyDataType_ELSIZE( descr ) ( ( descr ) -> elsize )
#define PyDataType_C_METADATA( descr ) ( ( descr ) -> c_metadata )

#endif

namespace csp::python
{

class AcquireGIL
{
public:
    AcquireGIL()  { m_state = PyGILState_Ensure(); }
    ~AcquireGIL() { PyGILState_Release( m_state ); }

private:
    PyGILState_STATE m_state;
};

class ReleaseGIL
{
public:
    ReleaseGIL()  { m_saveState = PyEval_SaveThread(); }
    ~ReleaseGIL() { PyEval_RestoreThread( m_saveState ); }

private:
    PyThreadState *m_saveState;
};

}

#endif
