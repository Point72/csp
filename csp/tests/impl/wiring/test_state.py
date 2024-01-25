import unittest
from datetime import datetime, timedelta

import csp
from csp import ts


class TestState(unittest.TestCase):
    def test_state_defined_in_with_block(self):
        @csp.node
        def test_node(in_: ts[bool]) -> ts[int]:
            with csp.state():
                x = 5

            if csp.ticked(in_):
                assert x == 5
                return x

        @csp.graph
        def test_graph() -> ts[int]:
            return test_node(csp.const(True))

        starttime = datetime(2021, 1, 1)

        ret = csp.run(test_graph, starttime=starttime, endtime=timedelta(seconds=15))

        self.assertEqual(ret[0][0][1], 5)


if __name__ == "__main__":
    unittest.main()
