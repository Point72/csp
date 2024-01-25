#ifndef _IN_CSP_PYTHON_PYCSPTYPE_H
#define _IN_CSP_PYTHON_PYCSPTYPE_H

#include <csp/engine/CspType.h>
#include <csp/python/PyObjectPtr.h>

static_assert( sizeof( csp::DialectGenericType ) == sizeof( csp::python::PyObjectPtr ) );
static_assert( alignof( csp::DialectGenericType ) == alignof( csp::python::PyObjectPtr ) );


//hook in fromCtype conversion
namespace csp
{
template<>
struct CspType::Type::fromCType<csp::python::PyObjectPtr>
{
    static constexpr csp::CspType::Type type = csp::CspType::Type::DIALECT_GENERIC;
};

}

#endif
