import unittest
from datetime import datetime, timedelta

import numpy

import csp


class TestCspCurve(unittest.TestCase):
    def test_basic(self):
        def g():
            return csp.curve(typ=float, data=[(csp.engine_start_time() + timedelta(i), i) for i in range(10)])

        res = csp.run(g, starttime=datetime(2020, 1, 1))[0]
        self.assertEqual(len(res), 10)
        self.assertEqual([v[0] for v in res], [datetime(2020, 1, 1) + timedelta(i) for i in range(10)])
        self.assertEqual([v[1] for v in res], list(range(10)))

    def test_timedelta(self):
        def g():
            return csp.curve(typ=float, data=[(timedelta(i), i) for i in range(10)])

        res = csp.run(g, starttime=datetime(2020, 1, 1))[0]
        self.assertEqual(len(res), 10)
        self.assertEqual([v[0] for v in res], [datetime(2020, 1, 1) + timedelta(i) for i in range(10)])
        self.assertEqual([v[1] for v in res], list(range(10)))

    def test_numpy(self):
        def g():
            times = numpy.array([csp.engine_start_time() + timedelta(i) for i in range(10)]).astype(numpy.datetime64)
            values = numpy.array(range(10))
            return csp.curve(typ=int, data=(times, values))

        res = csp.run(g, starttime=datetime(2020, 1, 1))[0]
        self.assertEqual(len(res), 10)
        self.assertEqual([v[0] for v in res], [datetime(2020, 1, 1) + timedelta(i) for i in range(10)])
        self.assertEqual([v[1] for v in res], list(range(10)))

    def test_empty_data(self):
        def g1():
            return csp.curve(typ=int, data=[])

        def g2():
            return csp.curve(typ=numpy.ndarray, data=[])

        def g3():
            return csp.curve(typ=str, data=(numpy.array([]), numpy.array([])))

        def g4():
            return csp.curve(typ=numpy.ndarray, data=(numpy.array([]), numpy.array([])))

        for g in [g1, g2, g3, g4]:
            res = csp.run(g, starttime=datetime(2020, 1, 1))[0]
            self.assertEqual(len(res), 0)


if __name__ == "__main__":
    unittest.main()
