from datetime import timedelta
from typing import Dict

from csp.impl.enum import Enum
from csp.impl.struct import Struct

CSP_AUTOGEN_HINTS = {"cpp_header": "csp/adapters/websocket/websocket_types.h"}


class WebsocketStatus(Enum):
    ACTIVE = 0
    GENERIC_ERROR = 1
    CONNECTION_FAILED = 2
    CLOSED = 3
    MESSAGE_SEND_FAIL = 4


class ActionType(Enum):
    CONNECT = 0
    DISCONNECT = 1
    PING = 2


class WebsocketHeaderUpdate(Struct):
    key: str
    value: str


class ConnectionRequest(Struct):
    uri: str
    action: ActionType = ActionType.CONNECT  # Connect, Disconnect, Ping, etc
    # Whetehr we maintain the connection
    persistent: bool = True  # Only relevant for Connect requests
    reconnect_interval: timedelta = timedelta(seconds=2)
    on_connect_payload: str = ""  # message to send on connect
    headers: Dict[str, str] = {}
