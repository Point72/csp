import unittest
from datetime import date, datetime, timedelta

import csp

try:
    from csp.adapters.perspective import PerspectiveAdapter
    from csp.impl.pandas_perspective import CspPerspectiveMultiTable, CspPerspectiveTable
    from csp.impl.perspective_common import PerspectiveWidget, is_perspective3

    HAS_PERSPECTIVE = True
except ImportError:
    HAS_PERSPECTIVE = False


class MyStruct(csp.Struct):
    my_str: str
    my_float: float
    my_bool: bool
    my_date: date
    my_datetime: datetime


def my_graph(output={}):
    adapter = PerspectiveAdapter(8000)
    table = adapter.create_table("Test")
    data = MyStruct(
        my_str="foo", my_float=1.0, my_bool=False, my_date=date(2020, 1, 1), my_datetime=datetime(2020, 1, 1)
    )
    table.publish(csp.unroll(csp.const([data, data])))
    output["table"] = table


class TestPerspectiveAdapter(unittest.TestCase):
    @unittest.skipIf(not HAS_PERSPECTIVE, "Test requires perspective")
    def test_adapter(self):
        output = {}
        csp.run(my_graph, output, starttime=datetime.utcnow(), endtime=timedelta(seconds=1))
