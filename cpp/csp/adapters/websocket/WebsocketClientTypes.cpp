#include "WebsocketClientTypes.h"

namespace csp {

INIT_CSP_ENUM( adapters::websocket::ClientStatusType,
               "ACTIVE",
               "GENERIC_ERROR",
               "CONNECTION_FAILED",
               "CLOSED",
               "MESSAGE_SEND_FAIL",
);

}