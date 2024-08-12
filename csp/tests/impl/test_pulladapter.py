import time
import unittest
from datetime import datetime, timedelta
from typing import List

import csp
from csp import PushMode, ts


class MyStruct(csp.Struct):
    a: int
    b: str


class TestPullAdapter(unittest.TestCase):
    def test_basic(self):
        # We just use the existing curve adapter to test pull since its currently implemented as a python PullInputAdapter
        @csp.node
        def check(burst: ts[List["T"]], lv: ts["T"], nc: ts["T"]):
            if csp.ticked(burst):
                self.assertEqual(len(burst), 2)
                self.assertEqual(burst[0], nc)

            if csp.ticked(nc):
                if isinstance(nc, int):
                    self.assertEqual(nc, csp.num_ticks(nc))
                else:
                    self.assertEqual(nc.a, csp.num_ticks(nc))
                if not csp.ticked(burst):
                    self.assertEqual(nc, burst[1])

            if csp.ticked(lv):
                if isinstance(lv, int):
                    self.assertEqual(lv, csp.num_ticks(lv) * 2)
                else:
                    self.assertEqual(lv.a, csp.num_ticks(lv) * 2)

        def graph(typ: type):
            raw_data = []
            td = timedelta()
            for x in range(1, 100, 2):
                if typ is int:
                    raw_data.append((td, x))
                    raw_data.append((td, x + 1))
                else:
                    raw_data.append((td, MyStruct(a=x, b=str(x))))
                    raw_data.append((td, MyStruct(a=x + 1, b=str(x + 1))))

                td += timedelta(seconds=1)

            nc = csp.curve(typ, raw_data, push_mode=csp.PushMode.NON_COLLAPSING)
            lv = csp.curve(typ, raw_data, push_mode=PushMode.LAST_VALUE)
            burst = csp.curve(typ, raw_data, push_mode=PushMode.BURST)
            check(burst, lv, nc)

        csp.run(graph, int, starttime=datetime(2023, 2, 21))
        # This was actually a bug specifically on Struct types: "PyPullAdapter crashes on burst of structs"
        csp.run(graph, MyStruct, starttime=datetime(2023, 2, 21))


if __name__ == "__main__":
    unittest.main()
