#pragma once

#include "csp/core/Enum.h"  // or whatever the correct path is

namespace csp::adapters::websocket {

struct WebsocketClientStatusTypeTraits
{
    enum _enum : unsigned char
    {
        ACTIVE = 0,
        GENERIC_ERROR = 1,
        CONNECTION_FAILED = 2,
        CLOSED = 3,
        MESSAGE_SEND_FAIL = 4,

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using ClientStatusType = Enum<WebsocketClientStatusTypeTraits>;

} 