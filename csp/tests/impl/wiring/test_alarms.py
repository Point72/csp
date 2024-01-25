import unittest
from datetime import datetime, timedelta

import csp
from csp import ts


class TestState(unittest.TestCase):
    def test_alarms_defined_in_with_block_scheduled_in_start_block(self):
        @csp.node
        def test_node() -> ts[int]:
            with csp.alarms():
                z = csp.alarm(bool)
            with csp.state():
                x = 5
            with csp.start():
                csp.schedule_alarm(z, timedelta(seconds=0), True)

            if csp.ticked(z):
                assert x == 5
                return x

        @csp.graph
        def test_graph() -> ts[int]:
            return test_node()

        starttime = datetime(2021, 1, 1)

        ret = csp.run(test_graph, starttime=starttime, endtime=timedelta(seconds=15))

        self.assertEqual(ret[0][0][1], 5)


if __name__ == "__main__":
    unittest.main()
