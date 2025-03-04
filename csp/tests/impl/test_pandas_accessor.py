import unittest
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

import csp
import csp.impl.pandas_accessor
from csp.impl.pandas_ext_type import TsDtype
from csp.impl.wiring.edge import Edge

_ = csp.impl.pandas_accessor  # To prevent IDE from removing import


def edge_eq(first, second):
    # Test that two edges are equal
    return first.run(starttime=datetime(2020, 1, 1), endtime=datetime(2020, 1, 2)) == second.run(
        starttime=datetime(2020, 1, 1), endtime=datetime(2020, 1, 2)
    )


class TestCspSeriesAccessor(unittest.TestCase):
    def setUp(self) -> None:
        idx = ["ABC", "DEF", "GJH"]
        self.series = pd.Series(
            [csp.const(99.0), csp.timer(timedelta(seconds=1), 103.0), np.nan], dtype=TsDtype(float), index=idx
        )
        self.empty = pd.Series([], dtype=TsDtype(float))
        multi = pd.MultiIndex(levels=[[], []], codes=[[], []])
        self.empty_multi = pd.Series([], index=multi, dtype=TsDtype(float))

    def test_apply(self):
        f = lambda x, y, z: x + y + z
        out = self.series.csp.apply(f, 1, z=2)
        for idx, edge in self.series.items():
            if pd.isnull(edge):
                self.assertTrue(pd.isnull(out[idx]))
            else:
                self.assertTrue(edge_eq(out[idx], edge.apply(f, 1, z=2)))

        out = self.empty.csp.apply(f, 1, z=2)
        self.assertEqual(out.dtype, self.empty.dtype)

    def test_pipe(self):
        trigger = csp.timer(timedelta(hours=1))
        out = self.series.csp.pipe((csp.sample, "x"), trigger=trigger)
        for idx, edge in self.series.items():
            if pd.isnull(edge):
                self.assertTrue(pd.isnull(out[idx]))
            else:
                self.assertTrue(edge_eq(out[idx], edge.pipe((csp.sample, "x"), trigger=trigger)))

        out = self.series.csp.pipe((csp.sample, "x"), trigger=trigger)
        self.assertEqual(out.dtype, self.empty.dtype)

    def test_binop(self):
        flag = pd.Series(
            [csp.const(True), csp.timer(timedelta(seconds=1), False), np.nan],
            dtype=TsDtype(bool),
            index=self.series.index,
        )
        out = flag.csp.binop(csp.filter, self.series)
        for idx, edge in self.series.items():
            if pd.isnull(edge):
                self.assertTrue(pd.isnull(out[idx]))
            else:
                self.assertTrue(edge_eq(out[idx], csp.filter(flag[idx], edge)))

        flag = flag.reset_index(drop=True)
        self.assertRaises(ValueError, flag.csp.binop, csp.filter, self.series)

        flag = pd.Series([], dtype=TsDtype(bool))
        out = flag.csp.binop(csp.filter, self.empty)
        self.assertEqual(out.dtype, flag.dtype)

    def test_sample(self):
        trigger = csp.timer(timedelta(hours=1))
        out1 = self.series.csp.sample(trigger)
        out2 = self.series.csp.sample(timedelta(hours=1))
        out3 = self.series.csp.sample(pd.Timedelta(hours=1))
        for idx, edge in self.series.items():
            if pd.isnull(edge):
                self.assertTrue(pd.isnull(out1[idx]))
                self.assertTrue(pd.isnull(out2[idx]))
                self.assertTrue(pd.isnull(out3[idx]))
            else:
                self.assertTrue(edge_eq(out1[idx], csp.sample(trigger, edge)))
                self.assertTrue(edge_eq(out2[idx], csp.sample(trigger, edge)))
                self.assertTrue(edge_eq(out3[idx], csp.sample(trigger, edge)))

        out = self.empty.csp.sample(trigger)
        self.assertEqual(out.dtype, self.empty.dtype)

    def test_flatten(self):
        class SubStruct(csp.Struct):
            fld: int = 0

        class MyStruct(csp.Struct):
            a: float
            b: str
            c: SubStruct

        series = pd.Series([csp.const(MyStruct(a=4.0, b="x", c=SubStruct()))], dtype=TsDtype(MyStruct), name="test")
        out = series.csp.flatten()
        self.assertEqual(list(out.columns), ["test a", "test b", "test c fld"])
        self.assertEqual(out["test a"].dtype, TsDtype(float))
        self.assertEqual(out["test b"].dtype, TsDtype(str))
        self.assertEqual(out["test c fld"].dtype, TsDtype(int))

        # Change delimeter
        out = series.csp.flatten(delim="_")
        self.assertEqual(list(out.columns), ["test_a", "test_b", "test_c_fld"])
        self.assertEqual(out["test_a"].dtype, TsDtype(float))
        self.assertEqual(out["test_b"].dtype, TsDtype(str))
        self.assertEqual(out["test_c_fld"].dtype, TsDtype(int))

        # Don't prepend name
        out = series.csp.flatten(prepend_name=False)
        self.assertEqual(list(out.columns), ["a", "b", "fld"])
        self.assertEqual(out["a"].dtype, TsDtype(float))
        self.assertEqual(out["b"].dtype, TsDtype(str))
        self.assertEqual(out["fld"].dtype, TsDtype(int))

        # Not recursive
        out = series.csp.flatten(recursive=False)
        self.assertEqual(list(out.columns), ["test a", "test b", "test c"])
        self.assertEqual(out["test a"].dtype, TsDtype(float))
        self.assertEqual(out["test b"].dtype, TsDtype(str))
        self.assertEqual(out["test c"].dtype, TsDtype(SubStruct))

        # No name provided
        series.name = None
        out = series.csp.flatten()
        self.assertEqual(list(out.columns), ["a", "b", "c fld"])
        self.assertEqual(out["a"].dtype, TsDtype(float))
        self.assertEqual(out["b"].dtype, TsDtype(str))
        self.assertEqual(out["c fld"].dtype, TsDtype(int))

    def test_flatten_empty(self):
        class SubStruct(csp.Struct):
            fld: int = 0

        class MyStruct(csp.Struct):
            a: float
            b: str
            c: SubStruct

        series = pd.Series([], dtype=TsDtype(MyStruct), name="test")
        out = series.csp.flatten()
        self.assertEqual(list(out.columns), ["test a", "test b", "test c fld"])
        self.assertEqual(out["test a"].dtype, TsDtype(float))
        self.assertEqual(out["test b"].dtype, TsDtype(str))
        self.assertEqual(out["test c fld"].dtype, TsDtype(int))

    def test_valid(self):
        out = csp.run(self.series.csp.valid, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))
        self.assertEqual(out, {0: [(datetime(2020, 1, 1, 0, 0, 1), True)]})

    def test_synchronize(self):
        x = csp.const(100.0)
        series = pd.Series(
            [x, csp.delay(x, timedelta(seconds=1)), np.nan, csp.delay(x, timedelta(seconds=2))], dtype=TsDtype(float)
        )
        out = series.csp.synchronize(timedelta(seconds=10)).csp.run(
            starttime=datetime(2020, 1, 1), endtime=timedelta(minutes=1)
        )
        target = pd.DatetimeIndex(["2020-01-01 00:00:02", "2020-01-01 00:00:02", "2020-01-01 00:00:02"])
        pd.testing.assert_index_equal(out.index.get_level_values(1), target)

        out = series.csp.synchronize(timedelta(seconds=1)).csp.run(
            starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10)
        )
        target = pd.DatetimeIndex(["2020-01-01 00:00:01", "2020-01-01 00:00:01", "2020-01-01 00:00:03"])
        pd.testing.assert_index_equal(out.index.get_level_values(1), target)

    def test_run(self):
        out = self.series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
            ]
        )
        target = pd.Series([99.0, 103.0, 103.0], index=idx)
        pd.testing.assert_series_equal(out, target)

        out = self.series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=3), tick_count=2)
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:03")),
            ]
        )
        target = pd.Series([99.0, 103.0, 103.0], index=idx)
        pd.testing.assert_series_equal(out, target)

        out = self.series.csp.run(
            starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=3), tick_history=timedelta(seconds=1)
        )
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:03")),
            ]
        )
        target = pd.Series([99.0, 103.0, 103.0], index=idx)
        pd.testing.assert_series_equal(out, target)

        out = self.series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(0))
        idx = pd.MultiIndex.from_tuples([("ABC", pd.Timestamp("2020-01-01"))])
        target = pd.Series([99.0], index=idx)
        pd.testing.assert_series_equal(out, target)

    def test_run_nodata(self):
        idx = ["ABC", "DEF", "GJH"]
        self.series = self.series[["DEF", "GJH"]]
        out = self.series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(0))
        self.assertEqual(len(out), 0)
        self.assertEqual(out.dtype, float)
        self.assertEqual(out.index.nlevels, 2)
        self.assertIsInstance(out.index.levels[-1], pd.DatetimeIndex)

    def test_run_empty(self):
        out = self.empty.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 0)
        self.assertEqual(out.index.nlevels, 2)
        self.assertIsInstance(out.index.levels[-1], pd.DatetimeIndex)

        out = self.empty_multi.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 0)
        self.assertEqual(out.index.nlevels, 3)
        self.assertIsInstance(out.index.levels[-1], pd.DatetimeIndex)

    def test_run_repeat(self):
        # Test that running with repeat timestamps works
        t = datetime(2020, 1, 1)
        data = csp.curve(float, [(t, 99.0), (t, 98.0)])
        self.series = pd.Series([data, data + 4, np.nan], dtype=TsDtype(float), index=self.series.index)
        out = self.series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=0))
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01")),
            ]
        )
        target = pd.Series([99.0, 98.0, 103.0, 102.0], index=idx)
        pd.testing.assert_series_equal(out, target)

    def test_snap(self):
        out = self.series.csp.snap(starttime=datetime(2020, 1, 1))
        idx = pd.Index(["ABC", "DEF"])
        target = pd.Series([99.0, 103.0], index=idx)
        pd.testing.assert_series_equal(out, target)

        out = self.series.csp.snap(timeout=timedelta(seconds=0.5), starttime=datetime(2020, 1, 1))
        idx = pd.Index(["ABC"])
        target = pd.Series([99.0], index=idx)
        pd.testing.assert_series_equal(out, target)

    def test_snap_nodata(self):
        self.series = self.series[["DEF", "GJH"]]
        out = self.series.csp.snap(timeout=timedelta(0))
        self.assertEqual(len(out), 0)
        self.assertEqual(out.dtype, float)
        self.assertEqual(out.index.dtype, self.series.index.dtype)

    def test_snap_empty(self):
        out = self.empty.csp.snap(starttime=datetime(2020, 1, 1))
        pd.testing.assert_index_equal(out.index, self.empty.index)

        out = self.empty_multi.csp.snap(starttime=datetime(2020, 1, 1))
        pd.testing.assert_index_equal(out.index, self.empty_multi.index)


class TestCspDataFrameAccessor(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = ["ABC", "DEF", "GJH"]
        bid = pd.Series(
            [csp.const(99.0), csp.timer(timedelta(seconds=1), 103.0), np.nan], dtype=TsDtype(float), index=self.idx
        )
        ask = pd.Series(
            [csp.const(100.0), csp.timer(timedelta(seconds=2), 104.0), csp.const(100.0)],
            dtype=TsDtype(float),
            index=self.idx,
        )
        sector = ["X", "Y", "X"]
        name = [s + " Corp" for s in self.idx]
        self.df = pd.DataFrame(
            {
                "name": name,
                "bid": bid,
                "ask": ask,
                "sector": sector,
            }
        )

    def test_static_frame(self):
        df_s = self.df.csp.static_frame()
        pd.testing.assert_frame_equal(df_s, self.df[["name", "sector"]])

    def test_ts_frame(self):
        df_s = self.df.csp.ts_frame()
        pd.testing.assert_index_equal(df_s.columns, self.df[["bid", "ask"]].columns)
        pd.testing.assert_index_equal(df_s.index, self.df[["bid", "ask"]].index)

    def test_run(self):
        out = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
                ("GJH", pd.Timestamp("2020-01-01")),
            ]
        )
        records = [
            ("ABC Corp", 99.0, 100.0, "X"),
            ("DEF Corp", 103.0, np.nan, "Y"),
            ("DEF Corp", 103.0, 104.0, "Y"),
            ("GJH Corp", np.nan, 100.0, "X"),
        ]
        target = pd.DataFrame.from_records(records, columns=["name", "bid", "ask", "sector"], index=idx)
        pd.testing.assert_frame_equal(out, target)

        out = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2), tick_count=1)
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
                ("GJH", pd.Timestamp("2020-01-01")),
            ]
        )
        records = [("ABC Corp", 99.0, 100.0, "X"), ("DEF Corp", 103.0, 104.0, "Y"), ("GJH Corp", np.nan, 100.0, "X")]
        target = pd.DataFrame.from_records(records, columns=["name", "bid", "ask", "sector"], index=idx)
        pd.testing.assert_frame_equal(out, target)

        # tick_history test
        out = self.df.csp.run(
            starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2), tick_history=timedelta(seconds=0.5)
        )
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
                ("GJH", pd.Timestamp("2020-01-01")),
            ]
        )
        records = [("ABC Corp", 99.0, 100.0, "X"), ("DEF Corp", 103.0, 104.0, "Y"), ("GJH Corp", np.nan, 100.0, "X")]
        target = pd.DataFrame.from_records(records, columns=["name", "bid", "ask", "sector"], index=idx)
        pd.testing.assert_frame_equal(out, target)

    def test_run_ts_only(self):
        out = self.df.csp.ts_frame().csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
                ("GJH", pd.Timestamp("2020-01-01")),
            ]
        )
        records = [(99.0, 100.0), (103.0, np.nan), (103.0, 104.0), (np.nan, 100.0)]
        target = pd.DataFrame.from_records(records, columns=["bid", "ask"], index=idx)
        pd.testing.assert_frame_equal(out, target)

    def test_run_static_only(self):
        out = self.df.csp.static_frame().csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_index_equal(out.columns, pd.Index(["name", "sector"]))
        self.assertEqual(out.index.nlevels, 2)
        self.assertIsInstance(out.index.levels[-1], pd.DatetimeIndex)

    def test_run_nodata(self):
        self.df = self.df.loc[["DEF"]]
        out = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=0))
        self.assertTrue(out.empty)
        pd.testing.assert_index_equal(out.columns, pd.Index(["name", "bid", "ask", "sector"]))
        self.assertEqual(out.index.nlevels, 2)
        self.assertIsInstance(out.index.levels[-1], pd.DatetimeIndex)

    def test_run_duplicate_static(self):
        idx = ["ABC", "DEF", "ABC"]
        self.df.index = idx
        self.assertRaises(ValueError, self.df.csp.run, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))

    def test_run_repeat(self):
        # Test that running with repeat timestamps fails (because two different series have different indices, and one
        # of those indices has duplicates)
        t = datetime(2020, 1, 1)
        bid = pd.Series(
            [csp.curve(float, [(t, 99.0), (t, 98.0)]), csp.curve(float, [(t, 103.0), (t, 102.0)]), np.nan],
            dtype=TsDtype(float),
            index=self.idx,
        )
        ask = pd.Series(
            [csp.const(100.0), csp.timer(timedelta(seconds=2), 104.0), csp.const(100.0)],
            dtype=TsDtype(float),
            index=self.idx,
        )
        self.df["bid"] = bid
        self.df["ask"] = ask

        out = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=0))
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01")),
                ("GJH", pd.Timestamp("2020-01-01")),
            ]
        )
        records = [
            ("ABC Corp", 99.0, 100.0, "X"),
            ("ABC Corp", 98.0, np.nan, "X"),
            ("DEF Corp", 103.0, np.nan, "Y"),
            ("DEF Corp", 102.0, np.nan, "Y"),
            ("GJH Corp", np.nan, 100.0, "X"),
        ]
        target = pd.DataFrame.from_records(records, columns=["name", "bid", "ask", "sector"], index=idx)
        pd.testing.assert_frame_equal(out, target)

    def test_run_types(self):
        class MyObject:
            pass

        self.idx = ["ABC", "DEF", "GJH"]
        s_str = pd.Series([csp.const("a") for _ in self.idx], dtype=TsDtype(str), index=self.idx)
        s_int = pd.Series([csp.const(0) for _ in self.idx], dtype=TsDtype(int), index=self.idx)
        s_float = pd.Series([csp.const(0.0) for _ in self.idx], dtype=TsDtype(float), index=self.idx)
        s_bool = pd.Series([csp.const(True) for _ in self.idx], dtype=TsDtype(bool), index=self.idx)
        s_date = pd.Series([csp.const(date.min) for _ in self.idx], dtype=TsDtype(date), index=self.idx)
        s_object = pd.Series([csp.const(MyObject()) for _ in self.idx], dtype=TsDtype(MyObject), index=self.idx)
        self.df = pd.DataFrame(
            {
                "s_str": s_str,
                "s_int": s_int,
                "s_float": s_float,
                "s_bool": s_bool,
                "s_date": s_date,
                "s_object": s_object,
            }
        )
        out = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=0))
        dtypes = pd.Series(
            # Note we use np.array([1]).dtype because on windows it defaults to int32, linux is int64
            [object, np.array([1]).dtype, np.float64, bool, object, object],
            index=["s_str", "s_int", "s_float", "s_bool", "s_date", "s_object"],
        )
        pd.testing.assert_series_equal(out.dtypes, dtypes)

    def test_snap(self):
        out = self.df.csp.snap(starttime=datetime(2020, 1, 1))
        records = [("ABC Corp", 99.0, 100.0, "X"), ("DEF Corp", 103.0, np.nan, "Y"), ("GJH Corp", np.nan, 100.0, "X")]
        target = pd.DataFrame.from_records(records, columns=["name", "bid", "ask", "sector"], index=self.idx)
        pd.testing.assert_frame_equal(out, target)

    def test_snap_ts_only(self):
        out = self.df.csp.ts_frame().csp.snap(starttime=datetime(2020, 1, 1))
        records = [(99.0, 100.0), (103.0, np.nan), (np.nan, 100.0)]
        target = pd.DataFrame.from_records(records, columns=["bid", "ask"], index=self.idx)
        pd.testing.assert_frame_equal(out, target)

    def test_snap_static_only(self):
        out = self.df.csp.static_frame().csp.snap(starttime=datetime(2020, 1, 1))
        pd.testing.assert_frame_equal(out, self.df.csp.static_frame().iloc[[]])

    def test_snap_nodata(self):
        self.df = self.df.loc[["DEF"]]
        out = self.df.csp.snap(timeout=timedelta(seconds=0))
        self.assertTrue(out.empty)
        pd.testing.assert_index_equal(out.columns, pd.Index(["name", "bid", "ask", "sector"]))
        self.assertEqual(out.index.nlevels, 1)
        self.assertEqual(out.index.dtype, self.df.index.dtype)

    def test_sample(self):
        trigger = csp.timer(timedelta(hours=1))
        out1 = self.df.csp.sample(trigger)
        out2 = self.df.csp.sample(timedelta(hours=1))
        out3 = self.df.csp.sample(pd.Timedelta(hours=1))
        target = self.df.copy()
        target["bid"] = target["bid"].csp.sample(trigger)
        target["ask"] = target["ask"].csp.sample(trigger)
        start = datetime(2020, 1, 1)
        end = datetime(2020, 1, 2)
        target = target.csp.run(start, end)
        pd.testing.assert_frame_equal(out1.csp.run(start, end), target)
        pd.testing.assert_frame_equal(out2.csp.run(start, end), target)
        pd.testing.assert_frame_equal(out3.csp.run(start, end), target)

        self.df.csp.sample(trigger, inplace=True)
        pd.testing.assert_frame_equal(self.df.csp.run(start, end), target)

    def test_collect(self):
        out = self.df.csp.collect()
        self.assertIsInstance(out, pd.Series)
        self.assertIsInstance(out.dtype, TsDtype)
        C = out.dtype.subtype  # Locally assign the dynamically created struct for convenience
        self.assertTrue(issubclass(C, csp.Struct))
        self.assertDictEqual(C.metadata(), {"ask": float, "bid": float})
        pd.testing.assert_index_equal(out.index, self.df.index)

        # Make sure it can actually run
        out = out.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
                ("GJH", pd.Timestamp("2020-01-01")),
            ]
        )
        records = [C(bid=99.0, ask=100.0), C(bid=103.0), C(bid=103.0, ask=104.0), C(ask=100.0)]
        target = pd.Series(records, index=idx)
        pd.testing.assert_series_equal(out, target)

        # Now try with an explicitly passed struct
        class MyStruct(csp.Struct):
            bid: float

        out = self.df.csp.collect(columns=["bid"], struct_type=MyStruct)
        self.assertIsInstance(out, pd.Series)
        self.assertEqual(out.dtype, TsDtype(MyStruct))
        out = out.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        idx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
            ]
        )
        records = [
            MyStruct(
                bid=99.0,
            ),
            MyStruct(bid=103.0),
            MyStruct(bid=103.0),
        ]
        target = pd.Series(records, index=idx)
        pd.testing.assert_series_equal(out, target)

    def test_collect_nested(self):
        # Create a data frame with many different types and nested structs
        self.df = None

        class SubStruct(csp.Struct):
            fld: int = 0

        class MyStruct(csp.Struct):
            a: float
            b: str
            c: SubStruct

        series = pd.Series([csp.const(MyStruct(a=4.0, b="x", c=SubStruct()))], dtype=TsDtype(MyStruct))
        df = series.csp.flatten()
        out = df.csp.collect()

        self.assertIsInstance(out, pd.Series)
        self.assertIsInstance(out.dtype, TsDtype)
        C = out.dtype.subtype  # Locally assign the dynamically created struct for convenience
        self.assertTrue(issubclass(C, csp.Struct))
        metadata = C.metadata()
        self.assertEqual(metadata.keys(), MyStruct.metadata().keys())
        self.assertEqual(metadata["a"], MyStruct.metadata()["a"])
        self.assertEqual(metadata["b"], MyStruct.metadata()["b"])
        self.assertDictEqual(metadata["c"].metadata(), SubStruct.metadata())

        # Make sure it can actually run
        out = out.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        idx = pd.MultiIndex.from_tuples([(0, pd.Timestamp("2020-01-01"))])
        target = pd.Series([C(a=4.0, b="x", c=metadata["c"](fld=0))], index=idx)
        pd.testing.assert_series_equal(out, target)

        # Test with explicitly passed struct and subset of fields
        out = df.csp.collect(columns=["a", "c fld"], struct_type=MyStruct)
        self.assertIsInstance(out, pd.Series)
        self.assertEqual(out.dtype, TsDtype(MyStruct))
        out = out.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))

        idx = pd.MultiIndex.from_tuples([(0, pd.Timestamp("2020-01-01"))])
        target = pd.Series([MyStruct(a=4.0, c=SubStruct(fld=0))], index=idx)
        pd.testing.assert_series_equal(out, target)

    def _performance_test(self, starttime, endtime, N, M, dt, offset, collect, seed=1234, profile=False):
        # N = number of symbols (rows)
        # M = number of fields
        # dt = frequency of updates
        # async = a small offset to the ticks of each field from the others.
        # When on-zero, this can really affect performance of the merged frame.
        import cProfile
        import io
        import pstats
        import random
        import time

        random.seed(seed)
        self.idx = [f"Symbol{i}" for i in range(N)]
        fields = {
            f"field_{i}": pd.Series(
                [csp.timer(dt + i * offset, 100.0 + i) for _ in self.idx], dtype=TsDtype(float), index=self.idx
            )
            for i in range(M)
        }
        sector = [random.choice(["X", "Y", "Z"]) for _ in self.idx]
        name = [s + " Corp" for s in self.idx]
        data = {
            "name": name,
            "sector": sector,
        }
        data.update(fields)
        self.df = pd.DataFrame(fields)
        if profile:
            pr = cProfile.Profile()
            pr.enable()
        start = time.time()
        out = self.df.csp.run(starttime=starttime, endtime=endtime, collect=collect)
        end = time.time()
        print(len(out), end - start)
        print(out.head())
        if profile:
            pr.disable()
            s = io.StringIO()
            sortby = "cumulative"
            ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
            ps.print_stats(0.1)
            ps.print_callers(0.1)
            print(s.getvalue())

    def xtest_run_performance(self):
        N = 5000
        M = 2
        dt = timedelta(minutes=1)
        offset = timedelta(seconds=0)
        starttime = datetime(2020, 1, 1, 9, 30)
        endtime = datetime(2020, 1, 1, 16)
        self._performance_test(starttime, endtime, N, M, dt, offset, collect=True, profile=False)

    def xtest_run_performance_async(self):
        N = 500
        M = 10
        dt = timedelta(minutes=1)
        offset = timedelta(seconds=1)
        starttime = datetime(2020, 1, 1, 9, 30)
        endtime = datetime(2020, 1, 1, 16)
        self._performance_test(starttime, endtime, N, M, dt, offset, collect=True, profile=False)


class TestToCspSeriesAccessor(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 00:00:01"), pd.Timestamp("2020-01-01 00:00:02")]
        )
        self.midx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:02")),
            ]
        )

    def test_to_csp_null(self):
        inp = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, float)
        self.assertEqual(edge.nodedef.__class__, csp.null_ts)

    def test_to_csp_float(self):
        inp = pd.Series([99.0, 102.0, 103.0], index=self.idx)
        edge = inp.to_csp()
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        target = [
            (datetime(2020, 1, 1, 0, 0), 99.0),
            (datetime(2020, 1, 1, 0, 0, 1), 102.0),
            (datetime(2020, 1, 1, 0, 0, 2), 103.0),
        ]
        self.assertListEqual(out, target)

        inp = pd.Series([99.0, np.nan, 103.0], index=self.idx)
        edge = inp.to_csp()
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        target = [
            (datetime(2020, 1, 1, 0, 0), 99.0),
            (datetime(2020, 1, 1, 0, 0, 1), np.nan),
            (datetime(2020, 1, 1, 0, 0, 2), 103.0),
        ]
        pd.testing.assert_frame_equal(pd.DataFrame.from_records(out), pd.DataFrame.from_records(target))

    def test_to_csp_float_drop_na(self):
        inp = pd.Series([99.0, np.nan, 103.0], index=self.idx)
        edge = inp.to_csp(drop_na=True)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        target = [(datetime(2020, 1, 1, 0, 0), 99.0), (datetime(2020, 1, 1, 0, 0, 2), 103.0)]
        self.assertListEqual(out, target)

    def test_to_csp_int(self):
        inp = pd.Series([1, 3, 2], index=self.idx, dtype=int)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, int)

        inp = pd.Series([None, 4, 5], index=self.idx, dtype=pd.Int64Dtype())
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, int)

    def test_to_csp_str(self):
        inp = pd.Series([None, "b", "c"], index=self.idx, dtype=object)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, str)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 2)

        inp = pd.Series([None, "b", "c"], index=self.idx, dtype=pd.StringDtype())
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, str)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 2)

    def test_to_csp_date(self):
        inp = pd.Series([None, date(2020, 1, 2), date(2020, 1, 3)], index=self.idx, dtype=object)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, date)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 2)

    def test_to_csp_bool(self):
        inp = pd.Series([True, False, None], index=self.idx, dtype=bool)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, bool)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 3)

        inp = pd.Series([True, False, None], index=self.idx, dtype=object)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, bool)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 2)

    def test_to_csp_datetime(self):
        inp = pd.Series([pd.NaT, datetime(2020, 1, 1), datetime(2020, 1, 2)], index=self.idx)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, datetime)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 2)

        inp = pd.Series([pd.NaT, datetime(2020, 1, 1), datetime(2020, 1, 2)], index=self.idx, dtype=object)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, datetime)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 2)

    def test_to_csp_custom(self):
        class A:
            pass

        inp = pd.Series([None, A(), A()], index=self.idx)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, A)
        out = edge.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        self.assertEqual(len(out), 2)

    def test_to_csp_object(self):
        class A:
            pass

        inp = pd.Series([4, None, "foo"], index=self.idx)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, object)

        inp = pd.Series([None, None, None], index=self.idx)
        edge = inp.to_csp()
        self.assertEqual(edge.tstype.typ, object)

    def test_to_csp_multiindex(self):
        inp = pd.Series([99.0, 103.0, 103.0], index=self.midx)
        series = inp.to_csp()
        self.assertEqual(series.dtype, TsDtype(float))
        out = series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_series_equal(out, inp)

        inp = pd.Series([99.0, np.nan, 103.0], index=self.midx)
        series = inp.to_csp()
        self.assertEqual(series.dtype, TsDtype(float))
        out = series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_series_equal(out, inp)

    def test_to_csp_multiindex_drop_na(self):
        inp = pd.Series([99.0, np.nan, 103.0], index=self.midx)
        series = inp.to_csp(drop_na=True)
        self.assertEqual(series.dtype, TsDtype(float))
        out = series.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_series_equal(out, inp.dropna())


class TestToCspFrameAccessor(unittest.TestCase):
    def setUp(self) -> None:
        self.idx = pd.DatetimeIndex(
            [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 00:00:01"), pd.Timestamp("2020-01-01 00:00:02")]
        )
        self.midx = pd.MultiIndex.from_tuples(
            [
                ("ABC", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01")),
                ("DEF", pd.Timestamp("2020-01-01 00:00:01")),
                ("GJH", pd.Timestamp("2020-01-01 00:00:02")),
            ]
        )
        bid = pd.Series([99.0, 103.0, np.nan, np.nan], index=self.midx)
        ask = pd.Series([100.0, np.nan, 104.0, 100.0], index=self.midx)
        sector = ["X", "Y", "Z", "X"]
        name = ["ABC Corp", "DEF Corp", "DEF Corp", "GJH Corp"]
        self.df = pd.DataFrame({"name": name, "bid": bid, "ask": ask, "sector": sector}, index=self.midx)

    def test_single_index(self):
        df = self.df.reset_index(level=0, drop=True)
        out = df.to_csp()
        self.assertIsInstance(out, dict)
        self.assertIsInstance(out["name"], Edge)
        self.assertEqual(out["name"].tstype.typ, str)
        self.assertIsInstance(out["bid"], Edge)
        self.assertEqual(out["bid"].tstype.typ, float)
        self.assertIsInstance(out["ask"], Edge)
        self.assertEqual(out["ask"].tstype.typ, float)
        self.assertIsInstance(out["sector"], Edge)
        self.assertEqual(out["sector"].tstype.typ, str)

        out = df.to_csp(columns=["bid", "ask"])
        self.assertIsInstance(out, dict)
        self.assertEqual(out["name"], "GJH Corp")
        self.assertIsInstance(out["bid"], Edge)
        self.assertEqual(out["bid"].tstype.typ, float)
        self.assertIsInstance(out["ask"], Edge)
        self.assertEqual(out["ask"].tstype.typ, float)
        self.assertEqual(out["sector"], "X")

        out = df.to_csp(columns=["bid", "ask"], agg=pd.Series.mode)
        self.assertIsInstance(out, dict)
        self.assertEqual(out["name"], "DEF Corp")
        self.assertEqual(out["sector"], "X")

    def test_multiindex(self):
        df = self.df.to_csp()
        np.testing.assert_array_equal(df.columns, self.df.columns)
        self.assertEqual(df["name"].dtype, TsDtype(str))
        self.assertEqual(df["bid"].dtype, TsDtype(float))
        self.assertEqual(df["ask"].dtype, TsDtype(float))
        self.assertEqual(df["sector"].dtype, TsDtype(str))
        out = df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_frame_equal(out, self.df)

        df = self.df.to_csp(columns=["bid", "ask"])
        np.testing.assert_array_equal(df.columns, self.df.columns)
        self.assertEqual(df["name"].tolist(), ["ABC Corp", "DEF Corp", "GJH Corp"])
        self.assertEqual(df["bid"].dtype, TsDtype(float))
        self.assertEqual(df["ask"].dtype, TsDtype(float))
        self.assertEqual(df["sector"].tolist(), ["X", "Z", "X"])
        out = df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_frame_equal(out[["bid", "ask"]], self.df[["bid", "ask"]])

        df = self.df.to_csp(columns=["bid", "ask"], agg="first")
        self.assertEqual(df["sector"].tolist(), ["X", "Y", "X"])  # Note it uses "Y" instead of Z"

        # In this corner case, no time series columns are selected
        # Also note that even though it asks for "first", pandas drops the NA when computing it.
        df = self.df.to_csp(columns=[], agg="first")
        self.assertEqual(df["bid"].dtype, float)
        self.assertEqual(df["ask"].dtype, float)
        self.assertEqual(df["sector"].tolist(), ["X", "Y", "X"])
        self.assertEqual(df["ask"].tolist(), [100.0, 104.0, 100.0])

    def test_multiindex_drop_na(self):
        df = self.df.to_csp(drop_na=True)
        np.testing.assert_array_equal(df.columns, self.df.columns)
        self.assertEqual(df["name"].dtype, TsDtype(str))
        self.assertEqual(df["bid"].dtype, TsDtype(float))
        self.assertEqual(df["ask"].dtype, TsDtype(float))
        self.assertEqual(df["sector"].dtype, TsDtype(str))

        out = df["ask"].csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_series_equal(out, self.df["ask"].dropna())

        out = df["bid"].csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_series_equal(out, self.df["bid"].dropna())

        out = df[["name", "sector"]].csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        pd.testing.assert_frame_equal(out, self.df[["name", "sector"]])
