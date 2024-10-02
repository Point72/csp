#ifndef _IN_CSP_PYTHON_COMMON_H
#define _IN_CSP_PYTHON_COMMON_H

#include <csp/core/Platform.h>
#include <Python.h>

#define IS_PRE_PYTHON_3_11 (PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 11) )

#define INIT_PYDATETIME if( !PyDateTimeAPI ) { PyDateTime_IMPORT; }

namespace csp::python
{

class CSP_PUBLIC AcquireGIL
{
public:
    AcquireGIL()  { m_state = PyGILState_Ensure(); }
    ~AcquireGIL() { PyGILState_Release( m_state ); }

private:
    PyGILState_STATE m_state;
};

class CSP_PUBLIC ReleaseGIL
{
public:
    ReleaseGIL()  { m_saveState = PyEval_SaveThread(); }
    ~ReleaseGIL() { PyEval_RestoreThread( m_saveState ); }

private:
    PyThreadState *m_saveState;
};

}

#endif
