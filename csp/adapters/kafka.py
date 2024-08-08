from datetime import datetime, timedelta
from enum import IntEnum
from typing import TypeVar, Union
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
from csp.lib import _kafkaadapterimpl

_ = BytesMessageProtoMapper, DateTimeType, JSONTextMessageMapper, RawBytesMessageMapper, RawTextMessageMapper
T = TypeVar("T")


class KafkaStatusMessageType(IntEnum):
    OK = 0
    MSG_DELIVERY_FAILED = 1
    MSG_SEND_ERROR = 2
    MSG_RECV_ERROR = 3
    GENERIC_ERROR = 4


class KafkaStartOffset(csp.Enum):
    EARLIEST = 1  # Replay all of history
    LATEST = 2  # Start from new msgs
    START_TIME = 3  # Start from csp run starttime


class KafkaAdapterManager:
    def __init__(
        self,
        broker,
        start_offset: Union[KafkaStartOffset, timedelta, datetime] = None,
        group_id: str = None,
        group_id_prefix: str = "",
        max_threads=4,
        max_queue_size=1000000,
        auth=False,
        security_protocol="SASL_SSL",
        sasl_kerberos_keytab="",
        sasl_kerberos_principal="",
        ssl_ca_location="",
        sasl_kerberos_service_name="kafka",
        rd_kafka_conf_options=None,
        debug: bool = False,
        poll_timeout: timedelta = timedelta(seconds=1),
    ):
        """
        :param broker - broker URL
        :param start_offset - signify where to start the stream playback from (defaults to KafkaStartOffset.LATEST ). Can be
                             one of the KafkaStartOffset enum types.
                             datetime - to replay from the given absolute time
                             timedelta - this will be taken as an absolute offset from starttime to playback from
        :param group_id - ( optional ) if set, this adapter will behave as a consume-once consumer.  start_offset may not be
                            set in this case since adapter will always replay from the last consumed offset.
        :param group_id_prefix - ( optional ) when not passing an explicit group_id, a prefix can be supplied that will be use to
                            prefix the UUID generated for the group_id
        """
        if group_id is not None and start_offset is not None:
            raise ValueError("start_offset is not supported when consuming with group_id")

        if not group_id:
            start_offset = start_offset if start_offset is not None else KafkaStartOffset.LATEST

        consumer_properties = {
            "group.id": group_id,
            # To get end of parition notification for live / not live flag
            "enable.partition.eof": "true",
        }

        producer_properties = {"queue.buffering.max.messages": str(max_queue_size)}

        conf_properties = {
            "bootstrap.servers": broker,
        }

        if group_id is None:
            # If user didnt request a group_id we dont commit to allow multiple consumers to get the same stream of data
            consumer_properties["group.id"] = group_id_prefix + str(uuid4())
            consumer_properties["enable.auto.commit"] = "false"
            consumer_properties["auto.commit.interval.ms"] = "0"
        else:
            consumer_properties["auto.offset.reset"] = "earliest"

        self._properties = {
            "start_offset": start_offset.value if isinstance(start_offset, KafkaStartOffset) else start_offset,
            "max_threads": max_threads,
            "poll_timeout": poll_timeout,
            "rd_kafka_conf_properties": conf_properties,
            "rd_kafka_consumer_conf_properties": consumer_properties,
            "rd_kafka_producer_conf_properties": producer_properties,
        }

        if auth:
            self._properties["rd_kafka_conf_properties"].update(
                {
                    "security.protocol": security_protocol,
                    "sasl.kerberos.keytab": sasl_kerberos_keytab,
                    "sasl.kerberos.principal": sasl_kerberos_principal,
                    "sasl.kerberos.service.name": sasl_kerberos_service_name,
                    "ssl.ca.location": ssl_ca_location,
                }
            )

        if debug:
            rd_kafka_conf_options = rd_kafka_conf_options.copy() if rd_kafka_conf_options else {}
            rd_kafka_conf_options["debug"] = "all"
            # Force start_offset to none so we dont block on pull adapter and let status msgs through
            self._properties["start_offset"] = None

        if rd_kafka_conf_options:
            if not isinstance(rd_kafka_conf_options, dict):
                raise TypeError("rd_kafka_conf_options must be a dict")
            self._properties["rd_kafka_conf_properties"].update(rd_kafka_conf_options)

    # reset_offset can be one of the following:
    # "smallest, earliest, beginning, largest, latest, end, error"
    def subscribe(
        self,
        ts_type: type,
        msg_mapper: MsgMapper,
        topic,
        # Leave key None to subscribe to all messages on the topic
        # Note that if you subscribe to all messages, they are always flagged as "live" and cant be replayed in engine time
        key=None,
        field_map: Union[dict, str] = None,
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
        properties["topic"] = topic
        properties["key"] = key or ""
        properties["field_map"] = field_map
        properties["meta_field_map"] = meta_field_map
        properties["adjust_out_of_order_time"] = adjust_out_of_order_time

        return _kafka_input_adapter_def(self, ts_type, properties, push_mode)

    def publish(self, msg_mapper: MsgMapper, topic: str, key: str, x: ts["T"], field_map: Union[dict, str] = None):
        if isinstance(field_map, str):
            field_map = {"": field_map}

        # TODO fix up this type stuff
        from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer

        ts_type = ContainerTypeNormalizer.normalized_type_to_actual_python_type(x.tstype.typ)

        if not field_map and issubclass(ts_type, csp.Struct):
            field_map = ts_type.default_field_map()

        properties = msg_mapper.properties.copy()
        properties["topic"] = topic
        properties["key"] = key
        properties["field_map"] = field_map

        return _kafka_output_adapter_def(self, x, ts_type, properties)

    def status(self, push_mode=csp.PushMode.NON_COLLAPSING):
        ts_type = Status
        return status_adapter_def(self, ts_type, push_mode)

    def _create(self, engine, memo):
        """method needs to return the wrapped c++ adapter manager"""
        return _kafkaadapterimpl._kafka_adapter_manager(engine, self._properties)


_kafka_input_adapter_def = input_adapter_def(
    "kafka_input_adapter",
    _kafkaadapterimpl._kafka_input_adapter,
    ts["T"],
    KafkaAdapterManager,
    typ="T",
    properties=dict,
)
_kafka_output_adapter_def = output_adapter_def(
    "kafka_output_adapter",
    _kafkaadapterimpl._kafka_output_adapter,
    KafkaAdapterManager,
    input=ts["T"],
    typ="T",
    properties=dict,
)
