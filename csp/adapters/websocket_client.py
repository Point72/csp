import typing
from datetime import timedelta
from enum import IntEnum
from typing import Dict, List

import csp
from csp import ts
from csp.adapters.status import Status
from csp.adapters.utils import (
    BytesMessageProtoMapper,
    DateTimeType,
    JSONTextMessageMapper,
    MsgMapper,
    RawBytesMessageMapper,
    RawTextMessageMapper,
)
from csp.impl.wiring import input_adapter_def, output_adapter_def, status_adapter_def
from csp.lib import _websocketsadapterimpl

_ = (
    BytesMessageProtoMapper,
    DateTimeType,
    JSONTextMessageMapper,
    RawBytesMessageMapper,
    RawTextMessageMapper,
)
T = typing.TypeVar("T")


class WebsocketsStatustype(IntEnum):
    ACTIVE = 0
    GENERIC_ERROR = 1
    CONNECTION_FAILED = 2
    CLOSED = 3


class WebsocketsAdapterManager:
    def __init__(
        self,
        uri: str,
        verbose_log: bool = False,
        reconnect_interval: timedelta = timedelta(seconds=2),
        headers: Dict[str, str] = None,
    ):
        """
        uri: str
            where to connect
        verbose_log: bool = False
            should the websocket client also log using the builtin
        reconnect_interval: timedelta = timedelta(seconds=2)
            time interval to wait before trying to reconnect (must be >= 1 second)
        headers: Dict[str, str] = None
            headers to apply to the request during the handshake
        """
        assert reconnect_interval >= timedelta(seconds=1)
        self._properties = dict(
            uri=uri,
            verbose_log=verbose_log,
            reconnect_interval=reconnect_interval,
            headers=headers if headers else {},
            use_tls=uri.startswith("wss"),
        )

    def subscribe(
        self,
        ts_type: type,
        msg_mapper: MsgMapper,
        field_map: typing.Union[dict, str] = None,
        meta_field_map: dict = None,
        push_mode: csp.PushMode = csp.PushMode.NON_COLLAPSING,
    ):
        field_map = field_map or {}
        meta_field_map = meta_field_map or {}
        if isinstance(field_map, str):
            field_map = {field_map: ""}

        if not field_map and issubclass(ts_type, csp.Struct):
            field_map = ts_type.default_field_map()

        properties = msg_mapper.properties.copy()
        properties["field_map"] = field_map
        properties["meta_field_map"] = meta_field_map

        return _websockets_input_adapter_def(self, ts_type, properties, push_mode)

    def send(self, x: ts["T"]):
        return _websockets_output_adapter_def(self, x)

    def update_headers(self, x: ts[List[str]]):
        return _websockets_header_update_adapter_def(self, x)

    def status(self, push_mode=csp.PushMode.NON_COLLAPSING):
        ts_type = Status
        return status_adapter_def(self, ts_type, push_mode)

    def _create(self, engine, memo):
        """method needs to return the wrapped c++ adapter manager"""
        return _websocketsadapterimpl._websockets_adapter_manager(engine, self._properties)


_websockets_input_adapter_def = input_adapter_def(
    "websockets_input_adapter",
    _websocketsadapterimpl._websockets_input_adapter,
    ts["T"],
    WebsocketsAdapterManager,
    typ="T",
    properties=dict,
)

_websockets_output_adapter_def = output_adapter_def(
    "websockets_output_adapter",
    _websocketsadapterimpl._websockets_output_adapter,
    WebsocketsAdapterManager,
    input=ts["T"],
)

_websockets_header_update_adapter_def = output_adapter_def(
    "websockets_header_update_adapter",
    _websocketsadapterimpl._websockets_header_update_adapter,
    WebsocketsAdapterManager,
    input=ts[List[str]],
)
