import os
from datetime import date, datetime, timedelta

import pytest

import csp
from csp import ts
from csp.adapters.kafka import (
    AvroMessageMapper,
    DateTimeType,
    JSONTextMessageMapper,
    KafkaAdapterManager,
    KafkaStartOffset,
    RawBytesMessageMapper,
    RawTextMessageMapper,
)

from .kafka_utils import _precreate_topic


class MyData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime
    date: date


class SubData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime
    date: date
    b2: bool
    i2: int
    d2: float
    s2: str
    dt2: datetime
    date2: date
    prop1: float
    prop2: str


class MetaTextStruct(csp.Struct):
    mapped_c: str


class MetaSubStruct(csp.Struct):
    mapped_b: MetaTextStruct


class MetaPubData(csp.Struct):
    mapped_a: MetaSubStruct
    mapped_count: int


class MetaSubData(csp.Struct):
    mapped_a: MetaSubStruct
    mapped_count: int
    mapped_partition: int
    mapped_offset: int
    mapped_live: bool
    mapped_timestamp: datetime


class TestKafka:
    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_metadata(self, kafkaadapter):
        def graph(count: int):
            msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)

            pub_field_map = {"mapped_a": {"a": {"mapped_b": {"b": {"mapped_c": "c"}}}}, "mapped_count": "count"}

            sub_field_map = {"a": {"mapped_a": {"b": {"mapped_b": {"c": "mapped_c"}}}}, "count": "mapped_count"}

            meta_field_map = {
                "partition": "mapped_partition",
                "offset": "mapped_offset",
                "live": "mapped_live",
                "timestamp": "mapped_timestamp",
            }

            topic = f"test.metadata.{os.getpid()}"
            _precreate_topic(topic)
            subKey = "foo"
            pubKey = ["mapped_a", "mapped_b", "mapped_c"]

            c = csp.count(csp.timer(timedelta(seconds=0.1)))
            t = csp.sample(c, csp.const("foo"))

            pubStruct = MetaPubData.collectts(
                mapped_a=MetaSubStruct.collectts(mapped_b=MetaTextStruct.collectts(mapped_c=t)), mapped_count=c
            )

            # csp.print('pub', pubStruct)
            kafkaadapter.publish(msg_mapper, topic, pubKey, pubStruct, field_map=pub_field_map)

            sub_data = kafkaadapter.subscribe(
                MetaSubData,
                msg_mapper,
                topic,
                subKey,
                field_map=sub_field_map,
                meta_field_map=meta_field_map,
                push_mode=csp.PushMode.NON_COLLAPSING,
            )

            csp.add_graph_output("sub_data", sub_data)
            # csp.print('sub', sub_data)
            # Wait for at least count ticks and until we get a live tick
            done_flag = csp.count(sub_data) >= count
            done_flag = csp.and_(done_flag, sub_data.mapped_live == True)  # noqa: E712
            stop = csp.filter(done_flag, done_flag)
            csp.stop_engine(stop)

        count = 5
        results = csp.run(graph, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)
        assert len(results["sub_data"]) >= 5
        print(results)
        for result in results["sub_data"]:
            assert result[1].mapped_partition >= 0
            assert result[1].mapped_offset >= 0
            assert result[1].mapped_live is not None
            assert result[1].mapped_timestamp < datetime.utcnow()
        assert results["sub_data"][-1][1].mapped_live

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_basic(self, kafkaadapter):
        @csp.node
        def curtime(x: ts[object]) -> ts[datetime]:
            if csp.ticked(x):
                return csp.now()

        def graph(symbols: list, count: int):
            b = csp.merge(
                csp.timer(timedelta(seconds=0.2), True),
                csp.delay(csp.timer(timedelta(seconds=0.2), False), timedelta(seconds=0.1)),
            )
            i = csp.count(csp.timer(timedelta(seconds=0.15)))
            d = csp.count(csp.timer(timedelta(seconds=0.2))) / 2.0
            s = csp.sample(csp.timer(timedelta(seconds=0.4)), csp.const("STRING"))
            dt = curtime(b)
            date_ts = csp.apply(dt, lambda x: x.date(), date)

            struct = MyData.collectts(b=b, i=i, d=d, s=s, dt=dt, date=date_ts)

            msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)

            struct_field_map = {"b": "b2", "i": "i2", "d": "d2", "s": "s2", "dt": "dt2", "date": "date2"}

            done_flags = []
            topic = f"mktdata.{os.getpid()}"
            _precreate_topic(topic)
            for symbol in symbols:
                kafkaadapter.publish(msg_mapper, topic, symbol, b, field_map="b")
                kafkaadapter.publish(msg_mapper, topic, symbol, i, field_map="i")
                kafkaadapter.publish(msg_mapper, topic, symbol, d, field_map="d")
                kafkaadapter.publish(msg_mapper, topic, symbol, s, field_map="s")
                kafkaadapter.publish(msg_mapper, topic, symbol, dt, field_map="dt")
                kafkaadapter.publish(msg_mapper, topic, symbol, date_ts, field_map="date")
                kafkaadapter.publish(msg_mapper, topic, symbol, struct, field_map=struct_field_map)

                # This isnt used to publish just to collect data for comparison at the end
                pub_data = SubData.collectts(
                    b=b,
                    i=i,
                    d=d,
                    s=s,
                    dt=dt,
                    date=date_ts,
                    b2=struct.b,
                    i2=struct.i,
                    d2=struct.d,
                    s2=struct.s,
                    dt2=struct.dt,
                    date2=struct.date,
                )
                csp.add_graph_output(f"pall_{symbol}", pub_data)

                # csp.print('status', kafkaadapter.status())

                sub_data = kafkaadapter.subscribe(
                    ts_type=SubData,
                    msg_mapper=msg_mapper,
                    topic=topic,
                    key=symbol,
                    push_mode=csp.PushMode.NON_COLLAPSING,
                )

                sub_data = csp.firstN(sub_data, count)

                csp.add_graph_output(f"sall_{symbol}", sub_data)

                done_flag = csp.count(sub_data) == count
                done_flag = csp.filter(done_flag, done_flag)
                done_flags.append(done_flag)

            stop = csp.and_(*done_flags)
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)

        symbols = ["AAPL", "MSFT"]
        count = 100
        results = csp.run(
            graph, symbols, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True
        )
        for symbol in symbols:
            pub = results[f"pall_{symbol}"]
            sub = results[f"sall_{symbol}"]

            assert len(sub) == count
            assert [v[1] for v in sub] == [v[1] for v in pub[:count]]

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_start_offsets(self, kafkaadapter, kafkabroker):
        topic = f"test_start_offsets.{os.getpid()}"
        _precreate_topic(topic)
        msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)
        count = 10

        # Prep the data first
        def pub_graph():
            i = csp.count(csp.timer(timedelta(seconds=0.1)))
            struct = MyData.collectts(i=i)
            kafkaadapter.publish(msg_mapper, topic, "AAPL", struct)
            stop = csp.count(struct) == count
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            # csp.print('pub', struct)

        csp.run(pub_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)

        # grab start/end times
        def get_times_graph():
            kafkaadapter = KafkaAdapterManager(broker=kafkabroker, start_offset=KafkaStartOffset.EARLIEST)
            data = kafkaadapter.subscribe(
                MyData,
                msg_mapper=msg_mapper,
                topic=topic,
                key="AAPL",
                meta_field_map={"timestamp": "dt"},
                push_mode=csp.PushMode.NON_COLLAPSING,
            )
            stop = csp.count(data) == count
            csp.stop_engine(csp.filter(stop, stop))
            csp.add_graph_output("data", data)

            # csp.print('sub', data)
            # csp.print('status', kafkaadapter.status())

        all_data = csp.run(get_times_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)[
            "data"
        ]
        min_time = all_data[0][1].dt

        def get_data(start_offset, expected_count):
            kafkaadapter = KafkaAdapterManager(broker=kafkabroker, start_offset=start_offset)
            data = kafkaadapter.subscribe(
                MyData,
                msg_mapper=msg_mapper,
                topic=topic,
                key="AAPL",
                meta_field_map={"timestamp": "dt"},
                push_mode=csp.PushMode.NON_COLLAPSING,
            )
            stop = csp.count(data) == expected_count
            csp.stop_engine(csp.filter(stop, stop))
            csp.add_graph_output("data", data)

            # csp.print('data', data)

        res = csp.run(
            get_data,
            KafkaStartOffset.EARLIEST,
            10,
            starttime=datetime.utcnow(),
            endtime=timedelta(seconds=30),
            realtime=True,
        )["data"]
        # print(res)
        # If we playback from earliest but start "now", all data should still arrive but as realtime ticks
        assert len(res) == 10

        res = csp.run(
            get_data,
            KafkaStartOffset.LATEST,
            1,
            starttime=datetime.utcnow(),
            endtime=timedelta(seconds=1),
            realtime=True,
        )["data"]
        assert len(res) == 0

        res = csp.run(
            get_data, KafkaStartOffset.START_TIME, 10, starttime=min_time, endtime=timedelta(seconds=30), realtime=True
        )["data"]
        assert len(res) == 10

        # Test sim playback time as well
        for t, v in res:
            assert t == v.dt

        stime = all_data[2][1].dt + timedelta(milliseconds=1)
        expected = [x for x in all_data if x[1].dt >= stime]
        res = csp.run(
            get_data, stime, len(expected), starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True
        )["data"]
        assert len(res) == len(expected)

        res = csp.run(
            get_data, timedelta(seconds=0), len(expected), starttime=stime, endtime=timedelta(seconds=30), realtime=True
        )["data"]
        assert len(res) == len(expected)

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_raw_pubsub(self, kafkaadapter):
        @csp.node
        def data(x: ts[object]) -> ts[bytes]:
            if csp.ticked(x):
                return str(csp.now())

        class SubData(csp.Struct):
            msg: bytes

        def graph(symbols: list, count: int):
            t = csp.timer(timedelta(seconds=0.1), True)
            d = data(t)

            msg_mapper = RawBytesMessageMapper()

            done_flags = []
            topic = f"test_str.{os.getpid()}"
            _precreate_topic(topic)
            for symbol in symbols:
                topic = f"test_str.{os.getpid()}"
                kafkaadapter.publish(msg_mapper, topic, symbol, d)
                csp.add_graph_output(f"pub_{symbol}", d)

                # csp.print('status', kafkaadapter.status())

                sub_data = kafkaadapter.subscribe(
                    ts_type=SubData,
                    msg_mapper=RawTextMessageMapper(),
                    field_map={"": "msg"},
                    topic=topic,
                    key=symbol,
                    push_mode=csp.PushMode.NON_COLLAPSING,
                )

                sub_data_bytes = kafkaadapter.subscribe(
                    ts_type=bytes,
                    msg_mapper=RawTextMessageMapper(),
                    field_map="",
                    topic=topic,
                    key=symbol,
                    push_mode=csp.PushMode.NON_COLLAPSING,
                )

                sub_data = csp.firstN(sub_data.msg, count)
                sub_data_bytes = csp.firstN(sub_data_bytes, count)

                # csp.print('sub', sub_data)
                csp.add_graph_output(f"sub_{symbol}", sub_data)
                csp.add_graph_output(f"sub_bytes_{symbol}", sub_data_bytes)

                done_flag = csp.count(sub_data) + csp.count(sub_data_bytes) == count * 2
                done_flag = csp.filter(done_flag, done_flag)
                done_flags.append(done_flag)

            stop = csp.and_(*done_flags)
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)

        symbols = ["AAPL", "MSFT"]
        count = 10
        results = csp.run(
            graph, symbols, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True
        )
        # print(results)
        for symbol in symbols:
            pub = results[f"pub_{symbol}"]
            sub = results[f"sub_{symbol}"]
            sub_bytes = results[f"sub_bytes_{symbol}"]

            assert len(sub) == count
            assert [v[1] for v in sub] == [v[1] for v in pub[:count]]
            assert [v[1] for v in sub_bytes] == [v[1] for v in pub[:count]]

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_invalid_topic(self, kafkaadapterkwargs):
        class SubData(csp.Struct):
            msg: str

        kafkaadapter1 = KafkaAdapterManager(**kafkaadapterkwargs)

        # Was a bug where engine would stall
        def graph_sub():
            # csp.print('status', kafkaadapter.status())
            return kafkaadapter1.subscribe(
                ts_type=SubData, msg_mapper=RawTextMessageMapper(), field_map={"": "msg"}, topic="foobar", key="none"
            )

        # With bug this would deadlock
        with pytest.raises(RuntimeError):
            csp.run(graph_sub, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)
        kafkaadapter2 = KafkaAdapterManager(**kafkaadapterkwargs)

        def graph_pub():
            msg_mapper = RawTextMessageMapper()
            kafkaadapter2.publish(msg_mapper, x=csp.const("heyyyy"), topic="foobar", key="test_key124")

        # With bug this would deadlock
        with pytest.raises(RuntimeError):
            csp.run(graph_pub, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_invalid_broker(self, kafkaadapterkwargs):
        dict_with_broker = kafkaadapterkwargs.copy()
        dict_with_broker["broker"] = "foobar"

        kafkaadapter1 = KafkaAdapterManager(**dict_with_broker)

        class SubData(csp.Struct):
            msg: str

        # Was a bug where engine would stall
        def graph_sub():
            return kafkaadapter1.subscribe(
                ts_type=SubData, msg_mapper=RawTextMessageMapper(), field_map={"": "msg"}, topic="foobar", key="none"
            )

        # With bug this would deadlock
        with pytest.raises(RuntimeError):
            csp.run(graph_sub, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)

        kafkaadapter2 = KafkaAdapterManager(**dict_with_broker)

        def graph_pub():
            msg_mapper = RawTextMessageMapper()
            kafkaadapter2.publish(msg_mapper, x=csp.const("heyyyy"), topic="foobar", key="test_key124")

        # With bug this would deadlock
        with pytest.raises(RuntimeError):
            csp.run(graph_pub, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_meta_field_map_tick_timestamp_from_field(self, kafkaadapterkwargs):
        class SubData(csp.Struct):
            msg: str
            dt: datetime

        kafkaadapter1 = KafkaAdapterManager(**kafkaadapterkwargs)

        def graph_sub():
            return kafkaadapter1.subscribe(
                ts_type=SubData,
                msg_mapper=RawTextMessageMapper(),
                meta_field_map={"timestamp": "dt"},
                topic="foobar",
                tick_timestamp_from_field="dt",
            )

        with pytest.raises(ValueError):
            csp.run(graph_sub, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)

    def test_conf_options(self):
        mgr = KafkaAdapterManager(
            "broker123",
            rd_kafka_conf_options={"test": "a"},
            rd_kafka_consumer_conf_options={"consumer_test": "b"},
            rd_kafka_producer_conf_options={"producer_test": "c"},
        )
        assert mgr._properties["rd_kafka_conf_properties"]["test"] == "a"
        assert mgr._properties["rd_kafka_consumer_conf_properties"]["consumer_test"] == "b"
        assert mgr._properties["rd_kafka_producer_conf_properties"]["producer_test"] == "c"

        pytest.raises(ValueError, KafkaAdapterManager, "broker123", rd_kafka_consumer_conf_options={"group.id": "b"})

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_push_mode(self, kafkaadapter, kafkabroker):
        class BasicData(csp.Struct):
            a: int
            b: bool

        topic = f"test_burst.{os.getpid()}"
        _precreate_topic(topic)
        msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)
        count = 10

        def pub_graph():
            i = csp.count(csp.timer(timedelta(seconds=0.1)))
            struct = BasicData.collectts(a=i, b=(i > 0))
            kafkaadapter.publish(msg_mapper, topic, "foo", struct)
            stop = csp.count(struct) == count
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)

        csp.run(pub_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=5), realtime=True)

        def sub_graph():
            kafkaadapter = KafkaAdapterManager(broker=kafkabroker, start_offset=KafkaStartOffset.EARLIEST)
            stop_flags = []
            for key, push_mode in (
                ("burst", csp.PushMode.BURST),
                ("last", csp.PushMode.LAST_VALUE),
            ):
                data = kafkaadapter.subscribe(
                    BasicData,
                    msg_mapper=msg_mapper,
                    topic=topic,
                    key="foo",
                    push_mode=push_mode,
                )
                csp.add_graph_output(key, data)
                stop_flags.append(csp.count(data) == 1)

            stop = csp.and_(*stop_flags)
            csp.stop_engine(csp.filter(stop, stop))

        res = csp.run(sub_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=5), realtime=True)
        burst = res["burst"]
        assert len(burst) == 1
        assert isinstance(burst[0][1], list)
        assert len(burst[0][1]) == 10
        assert all([rv == BasicData(a=i + 1, b=True) for i, rv in enumerate(burst[0][1])])

        last = res["last"]
        assert len(last) == 1
        assert isinstance(last[0][1], BasicData)
        assert last[0][1] == BasicData(a=count, b=True)

        # non-collapsing is tested in other tests


# Helper function for Avro tests
def assert_values_equal(actual, expected, field_name="value"):
    """Helper to compare values, handling floats with tolerance"""
    if isinstance(expected, float):
        assert abs(actual - expected) < 0.001, f"{field_name}: expected {expected}, got {actual}"
    else:
        assert actual == expected, f"{field_name}: expected {expected}, got {actual}"


class TestKafkaAvro:
    """Comprehensive tests for Avro message format in Kafka adapter"""

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_roundtrip_basic_types(self, kafkaadapter):
        """Test Avro round-trip with all basic types and value verification"""

        class TradeData(csp.Struct):
            symbol: str
            is_buy: bool
            quantity: int
            price: float
            trade_time: datetime
            trade_date: date

        avro_schema = """
        {
            "type": "record",
            "name": "TradeData",
            "fields": [
                {"name": "symbol", "type": "string"},
                {"name": "is_buy", "type": "boolean"},
                {"name": "quantity", "type": "long"},
                {"name": "price", "type": "double"},
                {"name": "trade_time", "type": "long"},
                {"name": "trade_date", "type": "int"}
            ]
        }
        """

        def graph(count: int):
            msg_mapper = AvroMessageMapper(avro_schema=avro_schema, datetime_type=DateTimeType.UINT64_MICROS)

            topic = f"avro_basic.{os.getpid()}"
            _precreate_topic(topic)

            c = csp.count(csp.timer(timedelta(seconds=0.1)))
            symbol = csp.sample(c, csp.const("AAPL"))
            is_buy = csp.apply(c, lambda x: x % 2 == 0, bool)
            quantity = csp.apply(c, lambda x: x * 100, int)
            price = csp.apply(c, lambda x: 100.0 + x * 0.5, float)
            trade_time = csp.apply(c, lambda x: datetime(2024, 1, 1, 10, 0, 0) + timedelta(seconds=x), datetime)
            trade_date = csp.apply(trade_time, lambda dt: dt.date(), date)

            pub_struct = TradeData.collectts(
                symbol=symbol,
                is_buy=is_buy,
                quantity=quantity,
                price=price,
                trade_time=trade_time,
                trade_date=trade_date,
            )

            kafkaadapter.publish(msg_mapper, topic, ["symbol"], pub_struct)

            sub_data = kafkaadapter.subscribe(
                TradeData, msg_mapper, topic, "AAPL", push_mode=csp.PushMode.NON_COLLAPSING
            )

            csp.add_graph_output("pub", pub_struct)
            csp.add_graph_output("sub", sub_data)

            done_flag = csp.count(sub_data) >= count
            stop = csp.filter(done_flag, done_flag)
            csp.stop_engine(stop)

        count = 10
        results = csp.run(graph, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)

        pub = results["pub"]
        sub = results["sub"]

        assert len(pub) == count
        assert len(sub) >= count

        for i in range(count):
            pub_data = pub[i][1]
            sub_data = sub[i][1]

            assert sub_data.symbol == pub_data.symbol == "AAPL"
            assert sub_data.is_buy == pub_data.is_buy
            assert sub_data.quantity == pub_data.quantity
            assert_values_equal(sub_data.price, pub_data.price, "price")
            assert sub_data.trade_date == pub_data.trade_date

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_roundtrip_complex_types(self, kafkaadapter):
        """Test Avro round-trip with arrays, nested structs, and enums"""

        class MyEnum(csp.Enum):
            OPTION_A = 1
            OPTION_B = 2
            OPTION_C = 3

        class NestedInfo(csp.Struct):
            category: str
            score: float

        class ComplexData(csp.Struct):
            key: str
            tags: [str]
            values: [int]
            prices: [float]
            info: NestedInfo
            status: MyEnum

        avro_schema = """
        {
            "type": "record",
            "name": "ComplexData",
            "fields": [
                {"name": "key", "type": "string"},
                {"name": "tags", "type": {"type": "array", "items": "string"}},
                {"name": "values", "type": {"type": "array", "items": "long"}},
                {"name": "prices", "type": {"type": "array", "items": "double"}},
                {"name": "info", "type": {
                    "type": "record",
                    "name": "NestedInfo",
                    "fields": [
                        {"name": "category", "type": "string"},
                        {"name": "score", "type": "double"}
                    ]
                }},
                {"name": "status", "type": {
                    "type": "enum",
                    "name": "MyEnum",
                    "symbols": ["OPTION_A", "OPTION_B", "OPTION_C"]
                }}
            ]
        }
        """

        def graph(count: int):
            msg_mapper = AvroMessageMapper(avro_schema=avro_schema, datetime_type=DateTimeType.UINT64_NANOS)

            topic = f"avro_complex.{os.getpid()}"
            _precreate_topic(topic)

            c = csp.count(csp.timer(timedelta(seconds=0.1)))
            key = csp.sample(c, csp.const("test_key"))
            tags = csp.apply(c, lambda x: [f"tag{i}" for i in range(x % 3 + 1)], [str])
            values = csp.apply(c, lambda x: list(range(1, x % 5 + 2)), [int])
            prices = csp.apply(c, lambda x: [i * 1.5 for i in range(1, x % 4 + 2)], [float])
            info_category = csp.sample(c, csp.const("A"))
            info_score = csp.apply(c, lambda x: x * 10.0, float)
            info = NestedInfo.collectts(category=info_category, score=info_score)

            # Cycle through enum values
            status_values = [MyEnum.OPTION_A, MyEnum.OPTION_B, MyEnum.OPTION_C]
            status = csp.apply(c, lambda x: status_values[(x - 1) % 3], MyEnum)

            pub_struct = ComplexData.collectts(
                key=key, tags=tags, values=values, prices=prices, info=info, status=status
            )

            kafkaadapter.publish(msg_mapper, topic, ["key"], pub_struct)

            sub_data = kafkaadapter.subscribe(
                ComplexData,
                msg_mapper,
                topic,
                "test_key",
                push_mode=csp.PushMode.NON_COLLAPSING,
            )

            csp.add_graph_output("pub", pub_struct)
            csp.add_graph_output("sub", sub_data)

            done_flag = csp.count(sub_data) >= count
            stop = csp.filter(done_flag, done_flag)
            csp.stop_engine(stop)

        count = 8
        results = csp.run(graph, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)

        pub = results["pub"]
        sub = results["sub"]

        assert len(pub) == count
        assert len(sub) >= count

        for i in range(count):
            pub_data = pub[i][1]
            sub_data = sub[i][1]

            assert sub_data.key == "test_key"
            assert len(sub_data.tags) == len(pub_data.tags)
            assert sub_data.tags == pub_data.tags
            assert len(sub_data.values) == len(pub_data.values)
            assert sub_data.values == pub_data.values
            assert len(sub_data.prices) == len(pub_data.prices)
            for j in range(len(sub_data.prices)):
                assert_values_equal(sub_data.prices[j], pub_data.prices[j], f"prices[{j}]")
            assert sub_data.info.category == pub_data.info.category
            assert_values_equal(sub_data.info.score, pub_data.info.score, "info.score")
            assert sub_data.status == pub_data.status

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_roundtrip_nullable_fields(self, kafkaadapter):
        """Test Avro round-trip with nullable (union) fields"""

        class OptionalData(csp.Struct):
            key: str
            required_value: int
            optional_count: int

        avro_schema = """
        {
            "type": "record",
            "name": "OptionalData",
            "fields": [
                {"name": "key", "type": "string"},
                {"name": "required_value", "type": "long"},
                {"name": "optional_count", "type": ["null", "long"], "default": null}
            ]
        }
        """

        def graph(count: int):
            msg_mapper = AvroMessageMapper(avro_schema=avro_schema, datetime_type=DateTimeType.UINT64_NANOS)

            topic = f"avro_nullable.{os.getpid()}"
            _precreate_topic(topic)

            # Use firstN to ensure exactly count values are produced
            c = csp.firstN(csp.count(csp.timer(timedelta(seconds=0.1))), count)
            key = csp.sample(c, csp.const("optional_key"))
            optional_count = csp.apply(c, lambda x: x * 10, int)

            pub_struct = OptionalData.collectts(key=key, required_value=c, optional_count=optional_count)

            kafkaadapter.publish(msg_mapper, topic, ["key"], pub_struct)

            sub_data = kafkaadapter.subscribe(
                OptionalData, msg_mapper, topic, "optional_key", push_mode=csp.PushMode.NON_COLLAPSING
            )

            csp.add_graph_output("pub", pub_struct)
            csp.add_graph_output("sub", sub_data)

            done_flag = csp.count(sub_data) >= count
            stop = csp.filter(done_flag, done_flag)
            csp.stop_engine(stop)

        count = 5
        results = csp.run(graph, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)

        pub = results["pub"]
        sub = results["sub"]

        assert len(pub) == count
        assert len(sub) >= count

        for i in range(count):
            pub_data = pub[i][1]
            sub_data = sub[i][1]

            assert sub_data.key == pub_data.key
            assert sub_data.required_value == pub_data.required_value
            assert sub_data.optional_count == pub_data.optional_count

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_field_mapping_roundtrip(self, kafkaadapter):
        """Test Avro round-trip with bidirectional field mapping"""

        class PubData(csp.Struct):
            internal_id: str
            internal_value: float
            internal_count: int

        class SubData(csp.Struct):
            external_id: str
            external_value: float
            external_count: int

        avro_schema = """
        {
            "type": "record",
            "name": "WireData",
            "fields": [
                {"name": "wire_id", "type": "string"},
                {"name": "wire_value", "type": "double"},
                {"name": "wire_count", "type": "long"}
            ]
        }
        """

        def graph(count: int):
            msg_mapper = AvroMessageMapper(avro_schema=avro_schema, datetime_type=DateTimeType.UINT64_NANOS)

            topic = f"avro_fieldmap.{os.getpid()}"
            _precreate_topic(topic)

            c = csp.count(csp.timer(timedelta(seconds=0.1)))
            internal_id = csp.sample(c, csp.const("mapped_key"))
            internal_value = csp.apply(c, lambda x: x * 2.5, float)
            internal_count = c

            pub_struct = PubData.collectts(
                internal_id=internal_id, internal_value=internal_value, internal_count=internal_count
            )

            # Publish: internal -> wire
            pub_field_map = {"internal_id": "wire_id", "internal_value": "wire_value", "internal_count": "wire_count"}
            kafkaadapter.publish(msg_mapper, topic, ["internal_id"], pub_struct, field_map=pub_field_map)

            # Subscribe: wire -> external
            sub_field_map = {"wire_id": "external_id", "wire_value": "external_value", "wire_count": "external_count"}
            sub_data = kafkaadapter.subscribe(
                SubData, msg_mapper, topic, "mapped_key", field_map=sub_field_map, push_mode=csp.PushMode.NON_COLLAPSING
            )

            csp.add_graph_output("pub", pub_struct)
            csp.add_graph_output("sub", sub_data)

            done_flag = csp.count(sub_data) >= count
            stop = csp.filter(done_flag, done_flag)
            csp.stop_engine(stop)

        count = 7
        results = csp.run(graph, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)

        pub = results["pub"]
        sub = results["sub"]

        assert len(pub) == count
        assert len(sub) >= count

        for i in range(count):
            pub_data = pub[i][1]
            sub_data = sub[i][1]

            # Values should match despite different field names
            assert sub_data.external_id == pub_data.internal_id == "mapped_key"
            assert_values_equal(sub_data.external_value, pub_data.internal_value, "value")
            assert sub_data.external_count == pub_data.internal_count == i + 1

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_datetime_wire_formats(self, kafkaadapter):
        """Test Avro with different datetime wire format configurations"""

        class TimeData(csp.Struct):
            key: str
            timestamp: datetime

        avro_schema = """
        {
            "type": "record",
            "name": "TimeData",
            "fields": [
                {"name": "key", "type": "string"},
                {"name": "timestamp", "type": "long"}
            ]
        }
        """

        def test_wire_format(wire_format: DateTimeType, count: int):
            """Helper to test a specific datetime wire format"""

            def graph():
                msg_mapper = AvroMessageMapper(avro_schema=avro_schema, datetime_type=wire_format)

                topic = f"avro_datetime_{wire_format.name}.{os.getpid()}"
                _precreate_topic(topic)

                # Use a fixed base timestamp for predictable testing
                base_time = datetime(2024, 1, 1, 12, 0, 0)
                # Use firstN to ensure exactly count values are produced
                c = csp.firstN(csp.count(csp.timer(timedelta(seconds=0.1))), count)
                key = csp.sample(c, csp.const("time_key"))
                timestamp = csp.apply(c, lambda x: base_time + timedelta(seconds=x), datetime)

                pub_struct = TimeData.collectts(key=key, timestamp=timestamp)

                kafkaadapter.publish(msg_mapper, topic, ["key"], pub_struct)

                sub_data = kafkaadapter.subscribe(
                    TimeData, msg_mapper, topic, "time_key", push_mode=csp.PushMode.NON_COLLAPSING
                )

                csp.add_graph_output("pub", pub_struct)
                csp.add_graph_output("sub", sub_data)

                done_flag = csp.count(sub_data) >= count
                stop = csp.filter(done_flag, done_flag)
                csp.stop_engine(stop)

            results = csp.run(graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)
            return results

        count = 5

        for wire_format in [
            DateTimeType.UINT64_NANOS,
            DateTimeType.UINT64_MICROS,
            DateTimeType.UINT64_MILLIS,
            DateTimeType.UINT64_SECONDS,
        ]:
            results = test_wire_format(wire_format, count)

            pub = results["pub"]
            sub = results["sub"]

            assert len(pub) == count
            assert len(sub) >= count

            # Verify datetime values match (with appropriate precision)
            for i in range(count):
                pub_time = pub[i][1].timestamp
                sub_time = sub[i][1].timestamp

                if wire_format == DateTimeType.UINT64_SECONDS:
                    # Precision loss expected - check within 1 second
                    assert abs((sub_time - pub_time).total_seconds()) < 1.0
                elif wire_format == DateTimeType.UINT64_MILLIS:
                    # Check within 1 millisecond
                    assert abs((sub_time - pub_time).total_seconds()) < 0.001
                elif wire_format == DateTimeType.UINT64_MICROS:
                    # Check within 1 microsecond
                    assert abs((sub_time - pub_time).total_seconds()) < 0.000001
                else:
                    # Should be exact
                    assert sub_time == pub_time

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_invalid_schema(self, kafkaadapter):
        """Test that invalid Avro schema raises proper error"""

        class TestStruct(csp.Struct):
            key: str
            value: int

        invalid_schema = """
        {
            "type": "record",
            "name": "BadSchema",
            "fields": [
                {"name": "field1", "type": "invalid_type"}
            ]
        }
        """

        def graph():
            msg_mapper = AvroMessageMapper(avro_schema=invalid_schema, datetime_type=DateTimeType.UINT64_NANOS)
            topic = f"avro_invalid.{os.getpid()}"
            _precreate_topic(topic)

            key = csp.const("test")
            value = csp.const(42)
            pub_struct = TestStruct.collectts(key=key, value=value)

            kafkaadapter.publish(msg_mapper, topic, ["key"], pub_struct)

        with pytest.raises(RuntimeError, match="Failed to parse Avro schema"):
            csp.run(graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_missing_schema(self, kafkaadapter):
        """Test that missing Avro schema raises proper error"""

        class TestStruct(csp.Struct):
            key: str
            value: int

        def graph():
            msg_mapper = AvroMessageMapper(avro_schema="", datetime_type=DateTimeType.UINT64_NANOS)
            topic = f"avro_missing.{os.getpid()}"
            _precreate_topic(topic)

            key = csp.const("test")
            value = csp.const(42)
            pub_struct = TestStruct.collectts(key=key, value=value)

            kafkaadapter.publish(msg_mapper, topic, ["key"], pub_struct)

        with pytest.raises(RuntimeError, match="'avro_schema' property is missing or empty"):
            csp.run(graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_avro_empty_arrays(self, kafkaadapter):
        """Test Avro round-trip with empty arrays"""

        class EmptyArrayData(csp.Struct):
            key: str
            tags: [str]
            values: [int]
            prices: [float]

        avro_schema = """
        {
            "type": "record",
            "name": "EmptyArrayData",
            "fields": [
                {"name": "key", "type": "string"},
                {"name": "tags", "type": {"type": "array", "items": "string"}},
                {"name": "values", "type": {"type": "array", "items": "long"}},
                {"name": "prices", "type": {"type": "array", "items": "double"}}
            ]
        }
        """

        def graph(count: int):
            msg_mapper = AvroMessageMapper(avro_schema=avro_schema, datetime_type=DateTimeType.UINT64_NANOS)

            topic = f"avro_empty_arrays.{os.getpid()}"
            _precreate_topic(topic)

            c = csp.firstN(csp.count(csp.timer(timedelta(seconds=0.1))), count)
            key = csp.sample(c, csp.const("empty_key"))
            tags = csp.apply(c, lambda x: [], [str])
            values = csp.apply(c, lambda x: [], [int])
            prices = csp.apply(c, lambda x: [], [float])

            pub_struct = EmptyArrayData.collectts(key=key, tags=tags, values=values, prices=prices)

            kafkaadapter.publish(msg_mapper, topic, ["key"], pub_struct)

            sub_data = kafkaadapter.subscribe(
                EmptyArrayData, msg_mapper, topic, "empty_key", push_mode=csp.PushMode.NON_COLLAPSING
            )

            csp.add_graph_output("pub", pub_struct)
            csp.add_graph_output("sub", sub_data)

            done_flag = csp.count(sub_data) >= count
            stop = csp.filter(done_flag, done_flag)
            csp.stop_engine(stop)

        count = 3
        results = csp.run(graph, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)

        pub = results["pub"]
        sub = results["sub"]

        assert len(pub) == count
        assert len(sub) >= count

        for i in range(count):
            pub_data = pub[i][1]
            sub_data = sub[i][1]

            assert sub_data.key == pub_data.key == "empty_key"
            assert len(sub_data.tags) == 0
            assert len(sub_data.values) == 0
            assert len(sub_data.prices) == 0
            assert sub_data.tags == pub_data.tags == []
            assert sub_data.values == pub_data.values == []
            assert sub_data.prices == pub_data.prices == []
