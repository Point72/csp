#pragma once

#include <csp/python/Conversions.h>

namespace csp::python 
{

    // Build a csp struct from a python dictionary.
    StructPtr structFromDict( const StructMetaPtr& struct_meta, PyObject* dict);

}