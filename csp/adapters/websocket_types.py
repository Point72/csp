from csp.impl.enum import Enum
from csp.impl.struct import Struct

CSP_AUTOGEN_HINTS = {"cpp_header": "csp/adapters/websocket/websocket_types.h"}


class WebsocketStatus(Enum):
    ACTIVE = 0
    GENERIC_ERROR = 1
    CONNECTION_FAILED = 2
    CLOSED = 3
    MESSAGE_SEND_FAIL = 4


class WebsocketHeaderUpdate(Struct):
    key: str
    value: str
