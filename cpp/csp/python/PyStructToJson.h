#ifndef _IN_CSP_PYSTRUCT_TOJSON_H
#define _IN_CSP_PYSTRUCT_TOJSON_H

#include <csp/python/Conversions.h>

namespace csp::python
{

// Entry point for converting structs into a json string
std::string structToJson( const StructPtr& struct_ptr, PyObject * callable );

}

#endif
