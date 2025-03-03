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
    # Whether we maintain the connection
    persistent: bool = True  # Only relevant for Connect requests
    reconnect_interval: timedelta = timedelta(seconds=2)
    on_connect_payload: str = ""  # message to send on connect
    headers: Dict[str, str] = {}


# Only used internally
class InternalConnectionRequest(Struct):
    host: str  # Hostname parsed from the URI
    port: str  # Port number for the connection (parsed and sanitized from URI)
    route: str  # Resource path from URI, defaults to "/" if empty
    uri: str  # Complete original URI string

    # Connection behavior
    use_ssl: bool  # Whether to use secure WebSocket (wss://)
    reconnect_interval: timedelta  # Time to wait between reconnection attempts
    persistent: bool  # Whether to maintain a persistent connection

    # Headers and payloads
    headers: str  # HTTP headers for the connection as json string
    on_connect_payload: str  # Message to send when connection is established

    # Connection metadata
    action: str  # Connection action type (Connect, Disconnect, Ping, etc)
    dynamic: bool  # Whether the connection is dynamic
    binary: bool  # Whether to use binary mode for the connection
