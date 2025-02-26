#ifndef _IN_CSP_ADAPTERS_UTILS_MESSAGEENUMS_H
#define _IN_CSP_ADAPTERS_UTILS_MESSAGEENUMS_H

#include <csp/core/Enum.h>

namespace csp::adapters::utils
{

struct DateTimeWireTypeTraits
{
    enum _enum : unsigned char
    {
        UNKNOWN = 0,
        UINT64_NANOS   = 1,
        UINT64_MICROS  = 2,
        UINT64_MILLIS  = 3,
        UINT64_SECONDS = 4,

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using DateTimeWireType = csp::Enum<DateTimeWireTypeTraits>;

};

#endif
