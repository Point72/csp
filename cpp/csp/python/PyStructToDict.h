#ifndef _IN_CSP_PYSTRUCT_TODICT_H
#define _IN_CSP_PYSTRUCT_TODICT_H

#include <csp/python/Conversions.h>

namespace csp::python
{

// Entry point for converting structs into a python dict
PyObjectPtr structToDict( const StructPtr& struct_ptr, PyObject * callable, bool preserve_enums );

}

#endif
