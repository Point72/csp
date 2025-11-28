import copy
import typing
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
    hash_mutable,
)
from csp.impl.wiring import ReplayMode, input_adapter_def, output_adapter_def, status_adapter_def
from csp.lib import _kafkaadapterimpl

_ = BytesMessageProtoMapper, DateTimeType, JSONTextMessageMapper, RawBytesMessageMapper, RawTextMessageMapper
T = TypeVar("T")


class KafkaStatusMessageType(IntEnum):
    OK = 0
    MSG_DELIVERY_FAILED = 1
    MSG_SEND_ERROR = 2
    MSG_RECV_ERROR = 3
    GENERIC_ERROR = 4


# Backward compatible
KafkaStartOffset = ReplayMode


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
        rd_kafka_consumer_conf_options=None,
        rd_kafka_producer_conf_options=None,
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

        consumer_properties = rd_kafka_consumer_conf_options.copy() if rd_kafka_consumer_conf_options else {}
        if {"group.id", "enable.partition.eof"}.intersection(consumer_properties.keys()):
            raise ValueError(
                "'group.id' and 'enable.partition.eof' are not settable with rd_kafka_consumer_conf_options "
            )
        consumer_properties["group.id"] = group_id
        # To get end of partition notification for live / not live flag
        consumer_properties["enable.partition.eof"] = "true"

        producer_properties = rd_kafka_producer_conf_options.copy() if rd_kafka_producer_conf_options else {}
        producer_properties["queue.buffering.max.messages"] = str(max_queue_size)

        conf_properties = {
            "bootstrap.servers": broker,
        }

        if group_id is None:
            consumer_properties["enable.auto.commit"] = "false"
            consumer_properties["auto.commit.interval.ms"] = "0"
        else:
            consumer_properties["auto.offset.reset"] = "earliest"

        self._group_id_prefix = group_id_prefix
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

        rd_kafka_conf_options = rd_kafka_conf_options.copy() if rd_kafka_conf_options else {}
        if debug:
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
        tick_timestamp_from_field: str = None,
        include_msg_before_start_time: bool = True,
    ):
        """
        Subscribe to a Kafka topic and map incoming messages to CSP timeseries.

        :param ts_type: The timeseries type to map the data to. Can be a csp.Struct or basic timeseries type
        :param msg_mapper: MsgMapper object to manage mapping message protocol to struct
        :param topic: Topic to subscribe to
        :param key: Key to subscribe to. If None, subscribes to all messages on the topic.
                    Note: In this "wildcard" mode, all messages will be marked as "live"
        :param field_map: Dictionary of {message_field: struct_field} mapping or string for single field mapping
        :param meta_field_map: Dictionary mapping kafka metadata to struct fields. Supported fields:
                    - "partition": Kafka partition number
                    - "offset": Message offset
                    - "live": Whether message is live or replay
                    - "timestamp": Kafka message timestamp
                    - "key": Message key
        :param push_mode: Mode for handling incoming messages (LAST_VALUE, NON_COLLAPSING, BURST)
        :param adjust_out_of_order_time: Allow out-of-order messages by forcing time to max(time, prev_time), only applies during sim replay.
        :param tick_timestamp_from_field: Override engine tick time using this struct field. Only applies during sim replay
        :param include_msg_before_start_time: Include messages from Kafka with times (either the time from Kafka or from the message as specified with `tick_timestamp_from_field`) before the engine start time.
        """
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
        properties["include_msg_before_start_time"] = include_msg_before_start_time
        if tick_timestamp_from_field is not None:
            if meta_field_map.get("timestamp") == tick_timestamp_from_field:
                raise ValueError(
                    f"Field '{tick_timestamp_from_field}' cannot be used for both timestamp extraction and meta field mapping"
                )
            properties["tick_timestamp_from_field"] = tick_timestamp_from_field

        return _kafka_input_adapter_def(self, ts_type, properties, push_mode)

    def publish(
        self,
        msg_mapper: MsgMapper,
        topic: str,
        key: typing.Union[str, typing.List],
        x: ts["T"],
        field_map: typing.Union[dict, str] = None,
    ):
        """
        :param msg_mapper - MsgMapper object to manage mapping struct to message protocol
        :param topic - topic to publish to
        :param key   - a string field of the struct type being published that will be used as the dynamic key to publish to.
                       key can also be a list of fields to reach into a nested struct field to be used as the key
        :param x     - timeseries of a Struct type to publish out
        :param field_map - option fieldmap from struct fieldname -> published field name
        """
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

    def __hash__(self):
        return hash((self._group_id_prefix, hash_mutable(self._properties)))

    def __eq__(self, other):
        return (
            isinstance(other, KafkaAdapterManager)
            and self._group_id_prefix == other._group_id_prefix
            and self._properties == other._properties
        )

    def _create(self, engine, memo):
        """method needs to return the wrapped c++ adapter manager"""
        # If user didnt request a group_id we dont commit to allow multiple consumers to get the same stream of data
        # We defer generation of unique group id to this point after properties can be sanely memoized
        properties = self._properties
        if properties["rd_kafka_consumer_conf_properties"]["group.id"] is None:
            properties = copy.deepcopy(properties)
            properties["rd_kafka_consumer_conf_properties"]["group.id"] = self._group_id_prefix + str(uuid4())
        return _kafkaadapterimpl._kafka_adapter_manager(engine, properties)


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
