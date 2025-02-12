#ifndef _IN_CSP_ENGINE_ENUMS_H
#define _IN_CSP_ENGINE_ENUMS_H

#include <csp/core/Enum.h>

namespace csp
{

// NOTE these must align with the python side Enum definition ///
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

//ReplayMode is used by PushPull adapters
struct ReplayModeTraits
{
     enum _enum : unsigned char
    {
        UNKNOWN        = 0,
        EARLIEST       = 1,  //Replay all available data
        LATEST         = 2,  //no replay at all, start from latest
        START_TIME     = 3,  //replay from engine start time

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using ReplayMode = Enum<ReplayModeTraits>;

}

#endif
