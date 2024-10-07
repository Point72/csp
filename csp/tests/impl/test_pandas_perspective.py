import numpy as np
import pandas as pd
import pyarrow as pa
import unittest
from datetime import date, datetime, timedelta
from packaging import version

import csp
import csp.impl.pandas_accessor
from csp.impl.pandas_ext_type import TsDtype

try:
    import ipywidgets
    import perspective

    from csp.impl.pandas_perspective import CspPerspectiveMultiTable, CspPerspectiveTable
except ImportError:
    raise unittest.SkipTest("skipping perspective tests")


def view_to_df(view):
    ipc_bytes = view.to_arrow()
    table = pa.ipc.open_stream(ipc_bytes).read_all()
    table = table.drop(["index"])  # TODO: not sure why this isnt dropped in one of the tests.
    df = pd.DataFrame(table.to_pandas())

    for column in df:
        if df[column].dtype == "datetime64[ms]":
            df[column] = df[column].astype("datetime64[ns]")
        elif df[column].dtype == "category":
            df[column] = df[column].cat.reorder_categories(df[column].cat.categories.sort_values())

    if df.index.dtype == "category":
        df.index = df.index.cat.reorder_categories(df.index.cat.categories.sort_values())

    return df


class TestCspPerspectiveTable(unittest.TestCase):
    def setUp(self) -> None:
        cat = pd.CategoricalDtype(["ABC", "DEF", "GJH"])
        self.idx = pd.Series(["ABC", "DEF", "GJH"], dtype=cat)
        bid = pd.Series(
            [csp.const(99.0), csp.timer(timedelta(seconds=1), 103.0), np.nan], dtype=TsDtype(float), index=self.idx
        )
        ask = pd.Series(
            [csp.const(100.0), csp.timer(timedelta(seconds=2), 104.0), csp.const(100.0)],
            dtype=TsDtype(float),
            index=self.idx,
        )
        sector = pd.Series(["X", "Y", "X"], dtype="category", index=self.idx)
        name = pd.Series([s + " Corp" for s in self.idx], dtype="category", index=self.idx)
        self.df = pd.DataFrame(
            {
                "name": name,
                "bid": bid,
                "ask": ask,
                "sector": sector,
            }
        )

    def check_table_history(self, table, target, index_col, time_col):
        df = table.to_df(strings_to_categorical=True)

        df = df.set_index([index_col, time_col])
        df.index.set_names([None, None], inplace=True)
        df = df.sort_index()
        df = df.convert_dtypes()
        target = target.convert_dtypes()
        pd.testing.assert_frame_equal(df, target)

    def test_not_running(self):
        table = CspPerspectiveTable(self.df)
        self.assertRaises(ValueError, table.stop)
        self.assertRaises(ValueError, table.join)

    def test_keep_history(self):
        target = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))

        table = CspPerspectiveTable(self.df)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        self.check_table_history(table, target, "index", "timestamp")

        table = CspPerspectiveTable(self.df, index_col="my_index", time_col="my_timestamp")
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        self.check_table_history(table, target, "my_index", "my_timestamp")

        table = CspPerspectiveTable(self.df, throttle=timedelta(seconds=10))
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        self.check_table_history(table, target, "index", "timestamp")

        table = CspPerspectiveTable(self.df, throttle=timedelta(seconds=0))
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        self.check_table_history(table, target, "index", "timestamp")

        self.assertRaises(ValueError, CspPerspectiveTable, self.df, time_col=None, keep_history=True)

    def test_run_multiple(self):
        table = CspPerspectiveTable(self.df)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        table.join()
        df1 = table.to_df()
        table.start(starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=2))
        table.join()
        df2 = table.to_df()
        self.assertEqual(len(df2), len(df1))
        table.start(starttime=datetime(2020, 1, 2), endtime=timedelta(seconds=2), clear=False)
        table.join()
        df2 = table.to_df()
        self.assertEqual(len(df2), 2 * len(df1))

    # DAVIS: Hitting the index bug (perspective bug #2756)
    def test_limit(self):
        table = CspPerspectiveTable(self.df, limit=3)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20))
        table.join()
        out = table.to_df()
        self.assertEqual(len(out), 3)

        target = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20))
        target.index.set_names(["index", "timestamp"], inplace=True)
        target = target.reset_index()
        target = target.sort_values(["timestamp", "index"]).tail(3)
        target = target.sort_values(["index", "timestamp"]).reset_index(drop=True).convert_dtypes()
        out = out.sort_values(["index", "timestamp"]).reset_index(drop=True).convert_dtypes()
        if version.parse(perspective.__version__) >= version.parse("1.0.3"):
            pd.testing.assert_frame_equal(out, target)
        pd.testing.assert_frame_equal(out, target)

        self.assertRaises(ValueError, CspPerspectiveTable, self.df, keep_history=False, limit=3)

    # DAVIS: perspective pyarrow output is using milliseconds
    #           psp-pandas, now built on pyarrow is using ms
    #           But this test is expected nanoseconds.
    def test_localize(self):
        # In this mode, the timestamp column should be in "local time", not utc
        self.df["datetime_col"] = pd.Series(
            [csp.const(datetime(2020, 1, 1, 1)), csp.const(datetime(2020, 1, 1, 2)), np.nan],
            dtype=TsDtype(datetime),
            index=self.idx,
        )
        table = CspPerspectiveTable(self.df, time_col="my_timestamp", keep_history=False, localize=True)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        out = table.to_df()

        utc = out["my_timestamp"]
        target = pd.Series(
            ["2020-01-01", "2020-01-01 00:00:04", "2020-01-01"], dtype="datetime64[ns]", name="my_timestamp"
        )
        pd.testing.assert_series_equal(utc, target)

        utc = out["datetime_col"]
        target = pd.Series(["2020-01-01 01", "2020-01-01 02", np.nan], dtype="datetime64[ns]", name="datetime_col")
        pd.testing.assert_series_equal(utc, target)

    def test_snap(self):
        index_col = "my_index"
        time_col = "my_timestamp"
        target = self.df.csp.run(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        target.index.set_names([index_col, time_col], inplace=True)
        target = target.reset_index(level=-1)
        target = target.groupby(level=0).ffill()
        target = target.groupby(level=0).last().reset_index().convert_dtypes()

        table = CspPerspectiveTable(self.df, index_col=index_col, time_col=time_col, keep_history=False)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        out = table.to_df().convert_dtypes()
        pd.testing.assert_frame_equal(out, target)

        table = CspPerspectiveTable(self.df, index_col=index_col, time_col=None, keep_history=False)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        out = table.to_df().convert_dtypes()
        pd.testing.assert_frame_equal(out, target.drop(columns=time_col))

    def test_run_types(self):
        s_str = pd.Series([csp.const("a") for _ in self.idx], dtype=TsDtype(str), index=self.idx)
        s_int = pd.Series([csp.const(0) for _ in self.idx], dtype=TsDtype(int), index=self.idx)
        s_float = pd.Series([csp.const(0.1) for _ in self.idx], dtype=TsDtype(float), index=self.idx)
        s_bool = pd.Series([csp.const(True) for _ in self.idx], dtype=TsDtype(bool), index=self.idx)
        s_date = pd.Series(
            [csp.const(date(year=1996, month=9, day=9)) for _ in self.idx], dtype=TsDtype(date), index=self.idx
        )
        self.df = pd.DataFrame({"s_str": s_str, "s_int": s_int, "s_float": s_float, "s_bool": s_bool, "s_date": s_date})
        table = CspPerspectiveTable(self.df)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))
        table.join()
        df = table.to_df().convert_dtypes()
        if version.parse(pd.__version__) >= version.parse("1.2.0"):
            floatDtype = pd.Float64Dtype()
        else:
            floatDtype = np.dtype("float64")

        dtypes = pd.Series(
            {
                "index": pd.CategoricalDtype(["ABC", "DEF", "GJH"]),
                "timestamp": np.dtype("datetime64[ns]"),
                "s_str": pd.CategoricalDtype(["a"]),
                "s_int": pd.Int32Dtype(),
                "s_float": floatDtype,
                "s_bool": pd.BooleanDtype(),
                "s_date": np.dtype("O"),
            }
        )
        pd.testing.assert_series_equal(df.dtypes, dtypes)

    # DAVIS: Hitting the index bug (perspective bug #2756)
    def test_run_historical(self):
        index_col = "my_index"
        time_col = "my_timestamp"
        table = CspPerspectiveTable(self.df, index_col=index_col, time_col=time_col)
        out = table.run_historical(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        target = table.to_df().sort_values([index_col, time_col]).reset_index(drop=True)
        pd.testing.assert_frame_equal(view_to_df(out.view()), target)
        table = CspPerspectiveTable(self.df, index_col=index_col, time_col=time_col, keep_history=False)
        out = table.run_historical(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=4))
        table.join()
        target = table.to_df()
        pd.testing.assert_frame_equal(view_to_df(out.view()), target)

        table = CspPerspectiveTable(self.df, index_col=index_col, time_col=time_col, limit=3)
        out = table.run_historical(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20))
        out = view_to_df(out.view())
        self.assertEqual(len(out), 3)

        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=20))
        table.join()
        target = table.to_df().sort_values([index_col, time_col]).reset_index(drop=True).tail(3)
        pd.testing.assert_frame_equal(out.sort_values([index_col, time_col]), target)

    def test_real_time(self):
        table = CspPerspectiveTable(self.df, keep_history=False)
        table.start(endtime=timedelta(seconds=0.2), realtime=True)
        table.join()
        self.assertEqual(len(table.to_df()), 3)
        self.assertEqual(len(table.to_df().dropna()), 1)

        # Start it again, but stop right away
        self.assertFalse(table.is_running())
        starttime = datetime.utcnow()
        endtime = starttime + timedelta(minutes=1)
        table.start(starttime=starttime, endtime=endtime, realtime=True)
        self.assertTrue(table.is_running())
        table.stop()
        self.assertFalse(table.is_running())
        self.assertLess(table.to_df()["timestamp"].max(), endtime - timedelta(seconds=2))

    def test_empty(self):
        table = CspPerspectiveTable(self.df.csp.static_frame(), keep_history=True)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        table.join()
        df1 = table.to_df()
        self.assertEqual(len(df1), 0)

        # With keep_history False, this should just be the static frame
        table = CspPerspectiveTable(self.df.csp.static_frame(), time_col=None, keep_history=False)
        table.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2))
        table.join()
        df2 = table.to_df()
        # this static_frame() has object dtypes
        self.df.csp.static_frame().reset_index().info(verbose=True)
        pd.testing.assert_frame_equal(df2, self.df.csp.static_frame().reset_index())

    def test_get_widget(self):
        table = CspPerspectiveTable(self.df, index_col="my_index", time_col="my_timestamp")
        widget = table.get_widget()
        self.assertIsInstance(widget, perspective.PerspectiveWidget)
        self.assertEqual(widget.columns, ["name", "bid", "ask", "sector"])
        self.assertEqual(
            widget.aggregates,
            {"name": "last by index", "bid": "last by index", "ask": "last by index", "sector": "last by index"},
        )
        self.assertEqual(widget.group_by, ["my_index", "my_timestamp"])
        self.assertEqual(widget.sort, [["my_timestamp", "desc"]])

        table = CspPerspectiveTable(self.df, index_col="my_index", time_col=None, keep_history=False)
        widget = table.get_widget()
        self.assertIsInstance(widget, perspective.PerspectiveWidget)
        layout = widget.save()
        self.assertEqual(layout["columns"], ["my_index", "name", "bid", "ask", "sector"])
        self.assertEqual(layout["aggregates"], {})
        self.assertEqual(layout["group_by"], [])
        self.assertEqual(layout["sort"], [])

        table = CspPerspectiveTable(self.df)
        widget = table.get_widget(sort=[["foo", "asc"]], theme="Material Dark")
        self.assertEqual(widget.sort, [["foo", "asc"]])
        self.assertEqual(widget.theme, "Material Dark")

    def test_multi_table(self):
        tables = {"test1": CspPerspectiveTable(self.df), "test2": CspPerspectiveTable(self.df, keep_history=False)}
        tables["test3"] = tables["test1"]
        multi = CspPerspectiveMultiTable(tables)
        self.assertEqual(multi.tables, tables)

        multi.start(starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2), clear=False)
        multi.join()
        self.assertEqual(len(multi["test1"].to_df()), 4)
        self.assertEqual(len(multi["test2"].to_df()), 3)
        self.assertEqual(len(multi["test3"].to_df()), len(multi["test1"].to_df()))

        multi.start(endtime=timedelta(seconds=2), realtime=True, clear=False)
        multi.stop()
        self.assertEqual(len(multi["test1"].to_df()), 6)  # From the two csp.const ticks
        self.assertEqual(len(multi["test2"].to_df()), 3)
        self.assertEqual(len(multi["test3"].to_df()), len(multi["test1"].to_df()))

    def test_multi_table_widget(self):
        tables = {"test1": CspPerspectiveTable(self.df), "test2": CspPerspectiveTable(self.df, keep_history=False)}
        tables["test3"] = tables["test1"]
        multi = CspPerspectiveMultiTable(tables)

        w = multi.get_widget("Tab")
        self.assertIsInstance(w, ipywidgets.Tab)
        self.assertEqual(w.get_title(0), "test1")
        self.assertEqual(w.get_title(1), "test2")
        self.assertEqual(w.get_title(2), "test3")
        self.assertEqual(len(w.children), 3)

        w = multi.get_widget("Accordion")
        self.assertIsInstance(w, ipywidgets.Accordion)
        self.assertEqual(w.get_title(0), "test1")
        self.assertEqual(w.get_title(1), "test2")
        self.assertEqual(w.get_title(2), "test3")
        self.assertEqual(len(w.children), 3)

        w = multi.get_widget("HBox")
        self.assertIsInstance(w, ipywidgets.HBox)
        self.assertEqual(len(w.children), 3)

        # Illustrate two named views on the same table (since table1 and table3 are the same)
        config = {"test1": dict(sort=[["foo", "asc"]]), "test3": dict(sort=[["bar", "desc"]])}
        w = multi.get_widget("Tab", config)
        self.assertIsInstance(w, ipywidgets.Tab)
        self.assertEqual(len(w.children), 2)
        self.assertEqual(w.children[0].sort, [["foo", "asc"]])
        self.assertEqual(w.children[1].sort, [["bar", "desc"]])
