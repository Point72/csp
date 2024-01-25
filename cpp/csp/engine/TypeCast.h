#ifndef _IN_CSP_ENGINE_TypeCast_H
#define _IN_CSP_ENGINE_TypeCast_H

#include <csp/core/Exception.h>
#include <cstdint>
#include <limits>

namespace csp
{

template< typename T, typename V, bool both_arithmetic = std::is_arithmetic<T>::value && std::is_arithmetic<V>::value >
struct RangeCheck
{
    static_assert( std::is_same<T, V>::value, "Unsupported conversion types" );
    static void verifyInRange( const V &v )
    {
    }
};

template< typename T, typename V >
struct RangeCheck<T, V, true>
{
    static void verifyInRange( const V &v )
    {
    }
};

template<>
struct RangeCheck<std::int64_t, std::uint64_t, true>
{
    static void verifyInRange( const std::uint64_t &v )
    {
        CSP_TRUE_OR_THROW( v <= std::uint64_t( std::numeric_limits<int64_t>::max()), RangeError,
                           "Trying to convert out of range value " << v << " to int64_t" );
    }
};

template< typename T, typename V >
inline typename std::decay<T>::type cast( const V &v )
{
    RangeCheck<typename std::decay<T>::type, typename std::decay<V>::type>::verifyInRange( v );
    return typename std::decay<T>::type( v );
}

}

#endif
