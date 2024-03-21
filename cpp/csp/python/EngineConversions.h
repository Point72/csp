#ifndef _IN_CSP_PYTHON_ENGINECONVERSIONS_H
#define _IN_CSP_PYTHON_ENGINECONVERSIONS_H

#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/TimeSeriesProvider.h>
#include <Python.h>

namespace csp::python
{

//timeserise helper method
PyObject * lastValueToPython( const csp::TimeSeriesProvider * ts );
PyObject * valueAtIndexToPython( const csp::TimeSeriesProvider * ts, int32_t index );

}

#endif
