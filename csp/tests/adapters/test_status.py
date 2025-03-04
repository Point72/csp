import os
from datetime import datetime, timedelta

import pytest

import csp
from csp import ts
from csp.adapters.kafka import DateTimeType, JSONTextMessageMapper, KafkaStatusMessageType
from csp.adapters.status import Level

from .kafka_utils import _precreate_topic


class SubData(csp.Struct):
    a: bool


class TestStatus:
    @pytest.mark.skipif(not os.environ.get("CSP_TEST_KAFKA"), reason="Skipping kafka adapter tests")
    def test_basic(self, kafkaadapter):
        topic = f"csp.unittest.{os.getpid()}"
        key = "test_status"

        def graph():
            # publish as string
            a = csp.timer(timedelta(seconds=1), "string")

            msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)

            kafkaadapter.publish(msg_mapper, topic, key, a, field_map="a")

            # subscribe as bool
            sub_data = kafkaadapter.subscribe(
                ts_type=SubData, msg_mapper=msg_mapper, topic=topic, key=key, push_mode=csp.PushMode.NON_COLLAPSING
            )
            status = kafkaadapter.status()

            csp.add_graph_output("sub_data", sub_data)
            csp.add_graph_output("status", status)

            # stop after first message
            done_flag = csp.count(status) == 1
            csp.stop_engine(done_flag)

        _precreate_topic(topic)
        results = csp.run(graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
        status = results["status"][0][1]
        assert status.status_code == KafkaStatusMessageType.MSG_RECV_ERROR
        assert status.level == Level.ERROR


if __name__ == "__main__":
    pytest.main()
