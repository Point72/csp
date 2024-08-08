#ifndef _IN_CSP_PYTHON_PYCSPTYPE_H
#define _IN_CSP_PYTHON_PYCSPTYPE_H

#include <csp/core/Platform.h>
#include <csp/engine/CspType.h>
#include <csp/python/PyObjectPtr.h>

static_assert( sizeof( csp::DialectGenericType ) == sizeof( csp::python::PyObjectPtr ) );
static_assert( alignof( csp::DialectGenericType ) == alignof( csp::python::PyObjectPtr ) );


//hook in fromCtype conversion
namespace csp
{
template<>
struct CspType::TypeTraits::fromCType<csp::python::PyObjectPtr>
{
    static constexpr csp::CspType::TypeTraits::_enum type = csp::CspType::TypeTraits::DIALECT_GENERIC;
};

}

#endif
