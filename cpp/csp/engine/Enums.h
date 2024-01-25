#ifndef _IN_CSP_ENGINE_ENUMS_H
#define _IN_CSP_ENGINE_ENUMS_H

#include <csp/core/Enum.h>

namespace csp
{

// NOTE this must align with the python side Enum definition ///
struct PushModeTraits
{
    enum _enum : unsigned char
    {
        UNKNOWN        = 0,
        LAST_VALUE     = 1,
        NON_COLLAPSING = 2,
        BURST          = 3,

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using PushMode = Enum<PushModeTraits>;

}

#endif
