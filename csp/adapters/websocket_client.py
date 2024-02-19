import typing
from enum import IntEnum
from typing import Dict

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
from csp.lib import _wsclientadapterimpl

_ = (
    BytesMessageProtoMapper,
    DateTimeType,
    JSONTextMessageMapper,
    RawBytesMessageMapper,
    RawTextMessageMapper,
)
T = typing.TypeVar("T")


class WSClientStatustype(IntEnum):
    ACTIVE = 0
    GENERIC_ERROR = 1
    CONNECTION_FAILED = 2


class WSClientAdapterManager:
    def __init__(
        self,
        uri: str,
        verbose_log: bool = False,
        reconnect_seconds: int = 2,
        headers: Dict[str, str] = None,
    ):
        """
        uri: str
            where to connect
        verbose_log: bool = False
            should the websocket client also log using the builtin
        reconnect_seconts: int = 2
            number of seconds to wait before reconnecting to server
        headers: Dict[str, str] = None
            headers to apply to the request during the handshake
        """
        self._properties = dict(
            uri=uri,
            verbose_log=verbose_log,
            reconnect_seconds=reconnect_seconds,
            headers=headers if headers else {},
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

        return _wsclient_input_adapter_def(self, ts_type, properties, push_mode)

    def send(self, x: ts["T"]):
        return _wsclient_output_adapter_def(self, x)

    def status(self, push_mode=csp.PushMode.NON_COLLAPSING):
        ts_type = Status
        return status_adapter_def(self, ts_type, push_mode)

    def _create(self, engine, memo):
        """method needs to return the wrapped c++ adapter manager"""
        return _wsclientadapterimpl._wsclient_adapter_manager(engine, self._properties)


_wsclient_input_adapter_def = input_adapter_def(
    "wsclient_input_adapter",
    _wsclientadapterimpl._wsclient_input_adapter,
    ts["T"],
    WSClientAdapterManager,
    typ="T",
    properties=dict,
)
_wsclient_output_adapter_def = output_adapter_def(
    "wsclient_output_adapter",
    _wsclientadapterimpl._wsclient_output_adapter,
    WSClientAdapterManager,
    input=ts["T"],
)
