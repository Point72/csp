from datetime import datetime, timedelta

import csp
from csp.adapters.utils import DateTimeType, JSONTextMessageMapper

__all__ = ("_precreate_topic",)


def _precreate_topic(adapter, topic):
    """Since we test against confluent kafka, just use the kafka rest addon"""

    def g():
        msg_mapper = JSONTextMessageMapper(datetime_type=DateTimeType.UINT64_MICROS)
        adapter.publish(msg_mapper, topic, "foo", csp.const("test"), field_map="a")

    csp.run(g, starttime=datetime.utcnow(), endtime=timedelta(), realtime=True)
