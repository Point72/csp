import os
from datetime import datetime, timedelta

import pytest

import csp
from csp import ts
from csp.adapters.kafka import KafkaAdapterManager, KafkaStartOffset
from csp.adapters.utils import DateTimeType, JSONTextMessageMapper, RawBytesMessageMapper, RawTextMessageMapper

from .kafka_utils import _precreate_topic


class MyData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime


class SubData(csp.Struct):
    b: bool
    i: int
    d: float
    s: str
    dt: datetime
    b2: bool
    i2: int
    d2: float
    s2: str
    dt2: datetime
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
        topic = f"test.metadata.{os.getpid()}"

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

            subKey = "foo"
            pubKey = ["mapped_a", "mapped_b", "mapped_c"]

            c = csp.count(csp.timer(timedelta(seconds=1)))
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

            # Wait for at least count ticks and until we get a live tick
            done_flag = csp.and_(csp.count(sub_data) >= count, sub_data.mapped_live == True)  # noqa: E712
            stop = csp.filter(done_flag, done_flag)
            csp.stop_engine(stop)

        # warm up the topic
        _precreate_topic(kafkaadapter, topic)
        results = csp.run(graph, 5, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
        assert len(results["sub_data"]) >= 5

        for result in results["sub_data"]:
            assert result[1].mapped_partition >= 0
            assert result[1].mapped_offset >= 0
            assert result[1].mapped_live is not None
            assert result[1].mapped_timestamp < datetime.utcnow()
        # first record should be non live
        assert results["sub_data"][0][1].mapped_live is False
        # last record should be live
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
            struct = MyData.collectts(b=b, i=i, d=d, s=s, dt=dt)

            msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)

            struct_field_map = {"b": "b2", "i": "i2", "d": "d2", "s": "s2", "dt": "dt2"}

            done_flags = []

            for symbol in symbols:
                kafkaadapter.publish(msg_mapper, topic, symbol, b, field_map="b")
                kafkaadapter.publish(msg_mapper, topic, symbol, i, field_map="i")
                kafkaadapter.publish(msg_mapper, topic, symbol, d, field_map="d")
                kafkaadapter.publish(msg_mapper, topic, symbol, s, field_map="s")
                kafkaadapter.publish(msg_mapper, topic, symbol, dt, field_map="dt")
                kafkaadapter.publish(msg_mapper, topic, symbol, struct, field_map=struct_field_map)

                # This isnt used to publish just to collect data for comparison at the end
                pub_data = SubData.collectts(
                    b=b, i=i, d=d, s=s, dt=dt, b2=struct.b, i2=struct.i, d2=struct.d, s2=struct.s, dt2=struct.dt
                )
                csp.add_graph_output(f"pall_{symbol}", pub_data)

                sub_data = kafkaadapter.subscribe(
                    ts_type=SubData,
                    msg_mapper=msg_mapper,
                    topic=topic,
                    key=symbol,
                    push_mode=csp.PushMode.NON_COLLAPSING,
                )
                csp.add_graph_output(f"sall_{symbol}", sub_data)

                done_flag = csp.count(sub_data) == count
                done_flag = csp.filter(done_flag, done_flag)
                done_flags.append(done_flag)

            stop = csp.and_(*done_flags)
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)

        topic = f"mktdata.{os.getpid()}"
        _precreate_topic(kafkaadapter, topic)
        symbols = ["AAPL", "MSFT"]
        count = 50
        results = csp.run(
            graph, symbols, count * 2, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True
        )
        for symbol in symbols:
            pub = results[f"pall_{symbol}"]
            sub = results[f"sall_{symbol}"]

            # limit by the last `count`
            sub = sub[-1 * count :]
            pub = pub[-1 * count :]

            assert len(sub) == count
            assert [v[1] for v in sub] == [v[1] for v in pub[-1 * count :]]

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_start_offsets(self, kafkaadapter, kafkabroker):
        topic = f"test_start_offsets.{os.getpid()}"
        _precreate_topic(kafkaadapter, topic)
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

        csp.run(pub_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)

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

        all_data = csp.run(get_times_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)[
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
            endtime=timedelta(seconds=10),
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
            get_data, KafkaStartOffset.START_TIME, 10, starttime=min_time, endtime=timedelta(seconds=10), realtime=True
        )["data"]
        assert len(res) == 10

        # Test sim playback time as well
        for t, v in res:
            assert t == v.dt

        stime = all_data[2][1].dt + timedelta(milliseconds=1)
        expected = [x for x in all_data if x[1].dt >= stime]
        res = csp.run(
            get_data, stime, len(expected), starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True
        )["data"]
        assert len(res) == len(expected)

        res = csp.run(
            get_data, timedelta(seconds=0), len(expected), starttime=stime, endtime=timedelta(seconds=10), realtime=True
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

        topic = f"test_str.{os.getpid()}"
        _precreate_topic(kafkaadapter, topic)

        symbols = ["AAPL", "MSFT"]
        count = 10
        results = csp.run(
            graph, symbols, count, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True
        )
        for symbol in symbols:
            pub = results[f"pub_{symbol}"]
            sub = results[f"sub_{symbol}"]
            sub_bytes = results[f"sub_bytes_{symbol}"]

            assert len(sub) == count
            assert [v[1] for v in sub] == [v[1] for v in pub[:count]]
            assert [v[1] for v in sub_bytes] == [v[1] for v in pub[:count]]

    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    @pytest.mark.skip(reason="Not working")
    def test_invalid_topic(self, kafkaadapternoautocreate):
        class SubData(csp.Struct):
            msg: str

        # Was a bug where engine would stall
        def graph_sub():
            # csp.print('status', kafkaadapter.status())
            return kafkaadapternoautocreate.subscribe(
                ts_type=SubData, msg_mapper=RawTextMessageMapper(), field_map={"": "msg"}, topic="foobar", key="none"
            )

        # With bug this would deadlock
        with pytest.raises(RuntimeError):
            csp.run(graph_sub, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)

        def graph_pub():
            msg_mapper = RawTextMessageMapper()
            kafkaadapternoautocreate.publish(msg_mapper, x=csp.const("heyyyy"), topic="foobar", key="test_key124")

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
    def test_meta_field_map_tick_timestamp_from_field(self, kafkaadapter):
        class SubData(csp.Struct):
            msg: str
            dt: datetime

        def graph_sub():
            return kafkaadapter.subscribe(
                ts_type=SubData,
                msg_mapper=RawTextMessageMapper(),
                meta_field_map={"timestamp": "dt"},
                topic="foobar",
                tick_timestamp_from_field="dt",
            )

        with pytest.raises(ValueError):
            csp.run(graph_sub, starttime=datetime.utcnow(), endtime=timedelta(seconds=2), realtime=True)
