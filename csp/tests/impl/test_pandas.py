import unittest
from datetime import datetime, timedelta

import pandas as pd
from numpy import nan

import csp
from csp.impl.pandas import make_pandas


class TestPandas(unittest.TestCase):
    def test_make_pandas_basic(self):
        # Basic test of trigger/tindex/window functionality
        start = datetime(2020, 1, 1)
        dt1 = timedelta(minutes=1)
        dt2 = dt1 * 2
        end = dt2 * 3
        x = csp.count(csp.timer(dt1))
        y = csp.count(csp.timer(dt2))
        out1 = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, 20, None), starttime=start, endtime=end)[0]
        out2 = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, 20, tindex=x), starttime=start, endtime=end)[0]
        out3 = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, 2, tindex=x), starttime=start, endtime=end)[0]
        for out in [out1, out2, out3]:
            self.assertEqual(len(out), 3)
            self.assertEqual(out[0][0], start + dt2)
            self.assertEqual(out[1][0], start + 2 * dt2)
            self.assertEqual(out[2][0], start + 3 * dt2)

        ## First Tick
        idx = pd.DatetimeIndex([start + dt2])
        target = pd.DataFrame({"x": [2], "y": [1]}, index=idx)
        pd.testing.assert_frame_equal(out1[0][1], target)
        pd.testing.assert_frame_equal(out2[0][1], target)
        pd.testing.assert_frame_equal(out3[0][1], target)

        ## Second Tick
        # Notes:
        # - pandas is a bit inconsistent with whether or not it sets freq on the DateTimeIndex, so we drop for comparison.
        # - when there is missing data in an integer column, it uses float NaN and hence the column becomes float type
        idx = pd.DatetimeIndex([start + dt2, start + dt2 + dt1, start + dt2 + 2 * dt1])
        target = pd.DataFrame({"x": [2, 3, 4], "y": [1.0, nan, 2.0]}, index=idx)
        out1[1][1].index.freq = None
        pd.testing.assert_frame_equal(out1[1][1], target)

        target = pd.DataFrame({"x": [2, 3, 4], "y": [1, 1, 2]}, index=idx)
        out2[1][1].index.freq = None
        pd.testing.assert_frame_equal(out2[1][1], target)
        out3[1][1].index.freq = None
        pd.testing.assert_frame_equal(out3[1][1], target.iloc[-2:])

        ## Third Tick
        idx = pd.DatetimeIndex(
            [start + dt2, start + dt2 + dt1, start + dt2 + 2 * dt1, start + dt2 + 3 * dt1, start + dt2 + 4 * dt1]
        )
        target = pd.DataFrame({"x": [2, 3, 4, 5, 6], "y": [1.0, nan, 2.0, nan, 3.0]}, index=idx)
        out1[2][1].index.freq = None
        pd.testing.assert_frame_equal(out1[2][1], target)

        target = pd.DataFrame({"x": [2, 3, 4, 5, 6], "y": [1, 1, 2, 2, 3]}, index=idx)
        out2[2][1].index.freq = None
        pd.testing.assert_frame_equal(out2[2][1], target)
        out3[2][1].index.freq = None
        pd.testing.assert_frame_equal(out3[2][1], target.iloc[-2:])

    def test_make_pandas_window(self):
        # Tests: window length
        start = datetime(2020, 1, 1)
        dt1 = timedelta(minutes=1)
        dt2 = dt1 * 2
        end = dt2 * 2
        x = csp.count(csp.timer(dt1))
        y = csp.count(csp.timer(dt2))
        # Because x and y are asynchronous, we grab latest two ticks of each
        # If tindex is provided, everything is synchronous, and will return last two rows
        out1 = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, 2, None), starttime=start, endtime=end)[0]
        # Test that a timedelta window works the same way
        out2 = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, dt1, None), starttime=start, endtime=end)[0]
        # Test a timedelta window that's too small
        out3 = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, dt1 / 2, None), starttime=start, endtime=end)[0]
        for out in [out1, out2, out3]:
            self.assertEqual(len(out), 2)
            self.assertEqual(out[0][0], start + dt2)
            self.assertEqual(out[1][0], start + 2 * dt2)

        ## First Tick
        idx = pd.DatetimeIndex([start + dt2])
        target = pd.DataFrame({"x": [2], "y": [1]}, index=idx)
        pd.testing.assert_frame_equal(out1[0][1], target)
        pd.testing.assert_frame_equal(out2[0][1], target)
        pd.testing.assert_frame_equal(out3[0][1], target)

        ## Second Tick
        # Notes:
        # - pandas is a bit inconsistent with whether or not it sets freq on the DateTimeIndex, so we drop for comparison.
        # - when there is missing data in an integer column, it uses float NaN and hence the column becomes float type
        idx = pd.DatetimeIndex([start + dt2, start + dt2 + dt1, start + dt2 + 2 * dt1])
        target = pd.DataFrame({"x": [nan, 3.0, 4.0], "y": [1.0, nan, 2.0]}, index=idx)
        out1[1][1].index.freq = None
        pd.testing.assert_frame_equal(out1[1][1], target)

        idx = pd.DatetimeIndex([start + dt2 + dt1, start + dt2 + 2 * dt1])
        target = pd.DataFrame({"x": [3, 4], "y": [nan, 2.0]}, index=idx)
        out2[1][1].index.freq = None
        pd.testing.assert_frame_equal(out2[1][1], target)

        idx = pd.DatetimeIndex([start + dt2 + 2 * dt1])
        target = pd.DataFrame({"x": [4], "y": [2]}, index=idx)
        out3[1][1].index.freq = None
        pd.testing.assert_frame_equal(out3[1][1], target)

    def test_make_pandas_init(self):
        # Test corner cases at the start (when not everything has ticked), and the wait_all_valid flag
        start = datetime(2020, 1, 1)
        dt1 = timedelta(minutes=1)
        dt2 = dt1 * 2
        end = dt2
        x = csp.count(csp.timer(dt1))
        y = csp.count(csp.timer(dt2))
        q = csp.timer(dt1 / 2)
        out1 = csp.run(
            lambda: make_pandas(q, {"x": x, "y": y}, 20, wait_all_valid=False), starttime=start, endtime=end
        )[0]
        out2 = csp.run(lambda: make_pandas(q, {"x": x, "y": y}, 20, wait_all_valid=True), starttime=start, endtime=end)[
            0
        ]

        ## out1
        self.assertEqual(len(out1), 4)
        self.assertEqual(out1[0][0], start + dt1 / 2)
        self.assertEqual(out1[1][0], start + dt1)
        self.assertEqual(out1[2][0], start + 3 * dt1 / 2)
        self.assertEqual(out1[3][0], start + 2 * dt1)

        idx = pd.DatetimeIndex([])
        target = pd.DataFrame({"x": pd.Series(dtype="int64"), "y": pd.Series(dtype="int64")}, index=idx)
        pd.testing.assert_frame_equal(out1[0][1], target)

        idx = pd.DatetimeIndex([start + dt1])
        target = pd.DataFrame({"x": [1], "y": [nan]}, index=idx)
        pd.testing.assert_frame_equal(out1[1][1], target)
        pd.testing.assert_frame_equal(out1[2][1], target)

        idx = pd.DatetimeIndex([start + dt1, start + dt2])
        target = pd.DataFrame({"x": [1, 2], "y": [nan, 1.0]}, index=idx)
        pd.testing.assert_frame_equal(out1[3][1], target)

        ## out2
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0][0], start + dt2)
        idx = pd.DatetimeIndex([start + dt2])
        target = pd.DataFrame({"x": [2], "y": [1]}, index=idx)
        out2[0][1].index.freq = None
        pd.testing.assert_frame_equal(out2[0][1], target)


class TestDataFrame(unittest.TestCase):
    def test_pandas_ts(self):
        start = datetime(2020, 1, 1)
        dt1 = timedelta(minutes=1)
        dt2 = dt1 * 2
        end = dt2 * 3
        x = csp.count(csp.timer(dt1))
        y = csp.count(csp.timer(dt2))
        df = csp.DataFrame({"x": x, "y": y})

        # Case 1: Simplest use case
        out = csp.run(lambda: df.to_pandas_ts("y", 2), starttime=start, endtime=end)[0]
        target = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, 2), starttime=start, endtime=end)[0]
        self.assertEqual(len(out), len(target))
        for i in range(len(out)):
            self.assertEqual(out[i][0], target[i][0])
            pd.testing.assert_frame_equal(out[i][1], target[i][1])

        # Case 2: Pass tindex
        out = csp.run(lambda: df.to_pandas_ts("y", 2, "x"), starttime=start, endtime=end)[0]
        target = csp.run(lambda: make_pandas(y, {"x": x, "y": y}, 2, tindex=x), starttime=start, endtime=end)[0]
        self.assertEqual(len(out), len(target))
        for i in range(len(out)):
            self.assertEqual(out[i][0], target[i][0])
            pd.testing.assert_frame_equal(out[i][1], target[i][1])

        # Case 3: Pass time series directly, and set wait_all_valid
        out = csp.run(lambda: df.to_pandas_ts(df["x"], 2, df["x"], wait_all_valid=False), starttime=start, endtime=end)[
            0
        ]
        target = csp.run(
            lambda: make_pandas(x, {"x": x, "y": y}, 2, tindex=x, wait_all_valid=False), starttime=start, endtime=end
        )[0]
        self.assertEqual(len(out), len(target))
        for i in range(len(out)):
            self.assertEqual(out[i][0], target[i][0])
            pd.testing.assert_frame_equal(out[i][1], target[i][1])

    def test_pandas_ts_types(self):
        start = datetime(2020, 1, 1)
        dt1 = timedelta(minutes=1)
        dt2 = dt1 * 2
        end = dt2 * 3
        x = csp.count(csp.timer(dt1))
        y = csp.count(csp.timer(dt2))
        df = csp.DataFrame({("A", "x"): x, ("B", "x"): csp.const("Test"), ("A", "y"): y, ("B", "y"): csp.const(start)})
        out = csp.run(lambda: df.to_pandas_ts(("A", "y"), 2, ("A", "x")), starttime=start, endtime=end)[0]
        self.assertEqual(len(out), 3)
        idx = pd.DatetimeIndex([start + dt2 + dt1, start + dt2 + 2 * dt1])
        target = pd.DataFrame(
            {("A", "x"): [3, 4], ("B", "x"): ["Test", "Test"], ("A", "y"): [1, 2], ("B", "y"): [start, start]},
            index=idx,
        )
        self.assertEqual(out[1][0], start + dt2 + 2 * dt1)
        pd.testing.assert_frame_equal(out[1][1], target)


if __name__ == "__main__":
    unittest.main()
