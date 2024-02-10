import typing
from datetime import datetime, timedelta
from enum import IntEnum
from uuid import uuid4

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
from csp.lib import _redisadapterimpl

_ = BytesMessageProtoMapper, DateTimeType, JSONTextMessageMapper, RawBytesMessageMapper, RawTextMessageMapper
T = typing.TypeVar("T")


class RedisStatusMessageType(IntEnum):
    OK = 0
    MSG_DELIVERY_FAILED = 1
    MSG_SEND_ERROR = 2
    MSG_RECV_ERROR = 3
    GENERIC_ERROR = 4


class RedisStartOffset(csp.Enum):
    EARLIEST = 1  # Replay all of history
    LATEST = 2  # Start from new msgs
    START_TIME = 3  # Start from csp run starttime


class RedisAdapterManager:
    def __init__(
        self,
        host: str,
        port: int,
        # connection options
        password: str = "",
        db: int = 0,
        keep_alive: bool = False,
        connect_timeout: timedelta = timedelta(milliseconds=100),
        socket_timeout: timedelta = timedelta(milliseconds=100),
        protocol: int = 2,
        # connection pool options
        pool_size: int = 1,
        pool_wait_timeout: timedelta = timedelta(milliseconds=100),
        pool_connection_lifetime: timedelta = timedelta(),
        pool_connection_idle_time: timedelta = timedelta(),
    ):
        self._properties = {
            "host": host,
            "port": port,
            "password": password,
            "db": db,
            "keep_alive": keep_alive,
            "connect_timeout": connect_timeout,
            "socket_timeout": socket_timeout,
            "resp": protocol,
            # pool options
            "pool_size": pool_size,
            "pool_wait_timeout": pool_wait_timeout,
            "pool_connection_lifetime": pool_connection_lifetime,
            "pool_connection_idle_time": pool_connection_idle_time,
            # TLS options
            # UNIX Domain Socket
        }

    def subscribe(
        self,
        ts_type: type,
        msg_mapper: MsgMapper,
        key: str,
        pattern: str = "",
        field_map: typing.Union[dict, str] = None,
        meta_field_map: dict = None,
        push_mode: csp.PushMode = csp.PushMode.LAST_VALUE,
        adjust_out_of_order_time: bool = False,
    ):
        field_map = field_map or {}
        meta_field_map = meta_field_map or {}
        if isinstance(field_map, str):
            field_map = {field_map: ""}

        if not field_map and issubclass(ts_type, csp.Struct):
            field_map = ts_type.default_field_map()

        properties = msg_mapper.properties.copy()
        properties["key"] = key
        properties["pattern"] = pattern
        properties["field_map"] = field_map
        properties["meta_field_map"] = meta_field_map
        properties["adjust_out_of_order_time"] = adjust_out_of_order_time
        return _redis_input_adapter_def(self, ts_type, properties, push_mode)

    def publish(
        self, msg_mapper: MsgMapper, key: str, x: ts["T"], field_map: typing.Union[dict, str] = None
    ):
        if isinstance(field_map, str):
            field_map = {"": field_map}

        # TODO fix up this type stuff
        from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
        ts_type = ContainerTypeNormalizer.normalized_type_to_actual_python_type(x.tstype.typ)

        if not field_map and issubclass(ts_type, csp.Struct):
            field_map = ts_type.default_field_map()

        properties = msg_mapper.properties.copy()
        properties["key"] = key
        properties["field_map"] = field_map
        return _redis_output_adapter_def(self, x, ts_type, properties)

    def status(self, push_mode=csp.PushMode.NON_COLLAPSING):
        ts_type = Status
        return status_adapter_def(self, ts_type, push_mode)

    def _create(self, engine, memo):
        """method needs to return the wrapped c++ adapter manager"""
        return _redisadapterimpl._redis_adapter_manager(engine, self._properties)


_redis_input_adapter_def = input_adapter_def(
    "redis_input_adapter",
    _redisadapterimpl._redis_input_adapter,
    ts["T"],
    RedisAdapterManager,
    typ="T",
    properties=dict,
)
_redis_output_adapter_def = output_adapter_def(
    "redis_output_adapter",
    _redisadapterimpl._redis_output_adapter,
    RedisAdapterManager,
    input=ts["T"],
    typ="T",
    properties=dict,
)
