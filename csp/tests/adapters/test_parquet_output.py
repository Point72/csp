"""Comprehensive tests for the parquet output adapter.

Tests all output scenarios: scalar types, structs, file rotation, split columns,
Arrow IPC, dict baskets, numpy arrays, batch sizes, compression, metadata, etc.
"""

import os
import tempfile
import unittest
from datetime import date, datetime, time, timedelta

import numpy
import pandas
import pyarrow
import pyarrow.ipc
import pyarrow.parquet
import pytz

import csp
from csp.adapters.output_adapters.parquet import ParquetOutputConfig
from csp.adapters.parquet import ParquetReader, ParquetWriter

START = datetime(2022, 1, 1, tzinfo=pytz.utc)


def _read_ipc(path):
    """Read an Arrow IPC stream fully into memory, leaving no open file handle.

    ``pyarrow.memory_map`` keeps an OS handle open until GC, which blocks
    ``tempfile.TemporaryDirectory`` cleanup on Windows ("file in use by another
    process"). Reading the bytes up front and parsing from an in-memory buffer
    avoids holding the file open.
    """
    with open(path, "rb") as fh:
        return pyarrow.ipc.open_stream(pyarrow.py_buffer(fh.read()))


class TestOutputScalarTypes(unittest.TestCase):
    """Test writing all scalar types to parquet and reading them back."""

    def _write_and_read_scalar(self, col_type, values, column_name="value"):
        """Helper: write a scalar column and read it back via pandas."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")
            curve_data = [(timedelta(seconds=i + 1), v) for i, v in enumerate(values)]

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish(column_name, csp.curve(col_type, curve_data))

            csp.run(g, starttime=START, endtime=timedelta(seconds=len(values) + 2))
            df = pandas.read_parquet(fname)
            return df

    def test_int(self):
        df = self._write_and_read_scalar(int, [1, 2, 3, -100, 0, 2**60])
        self.assertEqual(df["value"].tolist(), [1, 2, 3, -100, 0, 2**60])

    def test_float(self):
        df = self._write_and_read_scalar(float, [1.1, 2.2, 0.0, -3.14, float("inf")])
        self.assertAlmostEqual(df["value"].iloc[0], 1.1)
        self.assertAlmostEqual(df["value"].iloc[3], -3.14)
        self.assertEqual(df["value"].iloc[4], float("inf"))

    def test_bool(self):
        df = self._write_and_read_scalar(bool, [True, False, True, True, False])
        self.assertEqual(df["value"].tolist(), [True, False, True, True, False])

    def test_string(self):
        df = self._write_and_read_scalar(str, ["hello", "world", "", "with spaces", "unicode: 日本語"])
        self.assertEqual(df["value"].tolist(), ["hello", "world", "", "with spaces", "unicode: 日本語"])

    def test_datetime(self):
        dts = [datetime(2022, 1, i + 1, tzinfo=pytz.utc) for i in range(5)]
        df = self._write_and_read_scalar(datetime, dts)
        for i, dt in enumerate(dts):
            self.assertEqual(df["value"].iloc[i].to_pydatetime(), dt)

    def test_timedelta(self):
        tds = [timedelta(seconds=1), timedelta(minutes=5), timedelta(days=1), timedelta(microseconds=500)]
        df = self._write_and_read_scalar(timedelta, tds)
        for i, td in enumerate(tds):
            self.assertEqual(df["value"].iloc[i], td)

    def test_date(self):
        dates = [date(2022, 1, 1), date(2022, 6, 15), date(2023, 12, 31)]
        df = self._write_and_read_scalar(date, dates)
        for i, d in enumerate(dates):
            self.assertEqual(df["value"].iloc[i], d)

    def test_time(self):
        times = [time(9, 30, 0), time(12, 0, 0), time(23, 59, 59)]
        # Use csp reader for proper round-trip (pandas reads time64 differently)
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")
            curve_data = [(timedelta(seconds=i + 1), t) for i, t in enumerate(times)]

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("value", csp.curve(time, curve_data))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("value", reader.subscribe_all(time, "value"))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=10))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=10))
            read_times = [v[1] for v in res["value"]]
            self.assertEqual(read_times, times)

    def test_enum(self):
        class Color(csp.Enum):
            RED = 1
            GREEN = 2
            BLUE = 3

        values = [Color.RED, Color.GREEN, Color.BLUE, Color.RED]
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")
            curve_data = [(timedelta(seconds=i + 1), v) for i, v in enumerate(values)]

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("color", csp.curve(Color, curve_data))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("color", reader.subscribe_all(Color, "color"))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=10))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=10))
            self.assertEqual([v[1] for v in res["color"]], values)

    def test_bytes(self):
        values = [b"hello", b"\x00\x01\x02", b"", b"binary data"]
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")
            curve_data = [(timedelta(seconds=i + 1), v) for i, v in enumerate(values)]

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("data", csp.curve(bytes, curve_data))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("data", reader.subscribe_all(bytes, "data"))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=10))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=10))
            self.assertEqual([v[1] for v in res["data"]], values)

    def test_multiple_columns_different_types(self):
        """Write multiple columns of different types simultaneously."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("i", csp.curve(int, [(timedelta(seconds=1), 42), (timedelta(seconds=2), 99)]))
                writer.publish("f", csp.curve(float, [(timedelta(seconds=1), 3.14), (timedelta(seconds=2), 2.71)]))
                writer.publish("s", csp.curve(str, [(timedelta(seconds=1), "a"), (timedelta(seconds=2), "b")]))
                writer.publish("b", csp.curve(bool, [(timedelta(seconds=1), True), (timedelta(seconds=2), False)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertEqual(df["i"].tolist(), [42, 99])
            self.assertEqual(df["f"].tolist(), [3.14, 2.71])
            self.assertEqual(df["s"].tolist(), ["a", "b"])
            self.assertEqual(df["b"].tolist(), [True, False])

    def test_no_timestamp_column(self):
        """Write without a timestamp column."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, None, config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("x", csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(5)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=10))
            df = pandas.read_parquet(fname)
            self.assertNotIn("csp_timestamp", df.columns)
            self.assertEqual(df["x"].tolist(), list(range(5)))


class TestOutputStruct(unittest.TestCase):
    """Test struct publishing."""

    def test_basic_struct(self):
        class Trade(csp.Struct):
            price: float
            size: int
            exchange: str

        values = [
            Trade(price=100.5, size=10, exchange="NYSE"),
            Trade(price=101.0, size=20, exchange="ARCA"),
            Trade(price=99.5, size=15, exchange="NYSE"),
        ]

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish_struct(csp.curve(Trade, [(timedelta(seconds=i + 1), v) for i, v in enumerate(values)]))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("data", reader.subscribe_all(Trade))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=10))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=10))
            read_values = [v[1] for v in res["data"]]
            self.assertEqual(read_values, values)

    def test_struct_with_field_map(self):
        class Trade(csp.Struct):
            price: float
            size: int

        values = [Trade(price=100.0, size=10), Trade(price=200.0, size=20)]

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish_struct(
                    csp.curve(Trade, [(timedelta(seconds=i + 1), v) for i, v in enumerate(values)]),
                    field_map={"price": "trade_price", "size": "trade_size"},
                )

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertIn("trade_price", df.columns)
            self.assertIn("trade_size", df.columns)
            self.assertNotIn("price", df.columns)
            self.assertEqual(df["trade_price"].tolist(), [100.0, 200.0])
            self.assertEqual(df["trade_size"].tolist(), [10, 20])

    def test_wide_struct(self):
        class Wide(csp.Struct):
            f0: float
            f1: float
            f2: float
            f3: float
            f4: float
            f5: int
            f6: int
            f7: str
            f8: bool
            f9: float

        v = Wide(f0=1.0, f1=2.0, f2=3.0, f3=4.0, f4=5.0, f5=6, f6=7, f7="eight", f8=True, f9=10.0)

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish_struct(csp.const(v))

            csp.run(g, starttime=START, endtime=timedelta(seconds=2))
            df = pandas.read_parquet(fname)
            self.assertEqual(len(df), 1)
            self.assertEqual(df["f0"].iloc[0], 1.0)
            self.assertEqual(df["f7"].iloc[0], "eight")
            self.assertEqual(df["f8"].iloc[0], True)


class TestOutputFileRotation(unittest.TestCase):
    """Test file rotation via filename_provider."""

    def test_basic_rotation(self):
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g():
                data = csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(10)])
                rotate = csp.curve(
                    str,
                    [
                        (timedelta(seconds=5), os.path.join(d, "part1.parquet")),
                    ],
                )
                writer = ParquetWriter(
                    os.path.join(d, "part0.parquet"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    filename_provider=rotate,
                )
                writer.publish("value", data)

            csp.run(g, starttime=START, endtime=timedelta(seconds=15))
            df0 = pandas.read_parquet(os.path.join(d, "part0.parquet"))
            df1 = pandas.read_parquet(os.path.join(d, "part1.parquet"))
            self.assertEqual(df0["value"].tolist(), [0, 1, 2, 3])
            self.assertEqual(df1["value"].tolist(), [4, 5, 6, 7, 8, 9])

    def test_multiple_rotations(self):
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g():
                data = csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(9)])
                rotate = csp.curve(
                    str,
                    [
                        (timedelta(seconds=4), os.path.join(d, "p1.parquet")),
                        (timedelta(seconds=7), os.path.join(d, "p2.parquet")),
                    ],
                )
                writer = ParquetWriter(
                    os.path.join(d, "p0.parquet"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    filename_provider=rotate,
                )
                writer.publish("value", data)

            csp.run(g, starttime=START, endtime=timedelta(seconds=15))
            df0 = pandas.read_parquet(os.path.join(d, "p0.parquet"))
            df1 = pandas.read_parquet(os.path.join(d, "p1.parquet"))
            df2 = pandas.read_parquet(os.path.join(d, "p2.parquet"))
            self.assertEqual(df0["value"].tolist(), [0, 1, 2])
            self.assertEqual(df1["value"].tolist(), [3, 4, 5])
            self.assertEqual(df2["value"].tolist(), [6, 7, 8])

    def test_file_visitor_called(self):
        visited = []
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g():
                data = csp.curve(int, [(timedelta(seconds=1), 1), (timedelta(seconds=3), 2)])
                rotate = csp.curve(str, [(timedelta(seconds=2), os.path.join(d, "p1.parquet"))])
                writer = ParquetWriter(
                    os.path.join(d, "p0.parquet"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    filename_provider=rotate,
                    file_visitor=lambda f: visited.append(f),
                )
                writer.publish("value", data)

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            # file_visitor called for p0 when rotated, and p1 on stop
            self.assertEqual(len(visited), 2)
            self.assertTrue(visited[0].endswith("p0.parquet"))
            self.assertTrue(visited[1].endswith("p1.parquet"))

    def test_file_visitor_exception_surfaces_without_double_close(self):
        """If file_visitor raises, that error surfaces cleanly (no double-close crash on teardown)."""
        with tempfile.TemporaryDirectory() as d:

            def boom(_f):
                raise ValueError("visitor boom")

            @csp.graph
            def g():
                data = csp.curve(int, [(timedelta(seconds=1), 1), (timedelta(seconds=3), 2)])
                rotate = csp.curve(str, [(timedelta(seconds=2), os.path.join(d, "p1.parquet"))])
                writer = ParquetWriter(
                    os.path.join(d, "p0.parquet"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    filename_provider=rotate,
                    file_visitor=boom,
                )
                writer.publish("value", data)

            with self.assertRaisesRegex(Exception, "visitor boom"):
                csp.run(g, starttime=START, endtime=timedelta(seconds=5))

    def test_rotation_no_initial_file(self):
        """filename_provider only, no initial file_name."""
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g():
                data = csp.curve(int, [(timedelta(seconds=2), 1), (timedelta(seconds=3), 2)])
                rotate = csp.curve(str, [(timedelta(seconds=1), os.path.join(d, "output.parquet"))])
                writer = ParquetWriter(
                    None,
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    filename_provider=rotate,
                )
                writer.publish("value", data)

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(os.path.join(d, "output.parquet"))
            self.assertEqual(df["value"].tolist(), [1, 2])


class TestOutputArrowIPC(unittest.TestCase):
    """Test Arrow IPC (binary arrow) output format."""

    def test_basic_ipc(self):
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.arrow")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, write_arrow_binary=True, compression="")
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=i + 1), i * 10) for i in range(5)]))
                writer.publish("y", csp.curve(float, [(timedelta(seconds=i + 1), i * 1.5) for i in range(5)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=10))
            reader = _read_ipc(fname)
            table = reader.read_all()
            self.assertEqual(table.column("x").to_pylist(), [0, 10, 20, 30, 40])
            self.assertEqual(table.column("y").to_pylist(), [0.0, 1.5, 3.0, 4.5, 6.0])

    def test_ipc_with_compression(self):
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.arrow")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, write_arrow_binary=True, compression="zstd")
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 42)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            reader = _read_ipc(fname)
            table = reader.read_all()
            self.assertEqual(table.column("x").to_pylist(), [42])

    def test_ipc_round_trip_via_csp(self):
        """Write IPC and read back via csp ParquetReader."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.arrow")
            values = [10, 20, 30]

            @csp.graph
            def g_write():
                config = ParquetOutputConfig(allow_overwrite=True, write_arrow_binary=True, compression="")
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=i + 1), v) for i, v in enumerate(values)]))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp", binary_arrow=True)
                csp.add_graph_output("x", reader.subscribe_all(int, "x"))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=10))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=10))
            self.assertEqual([v[1] for v in res["x"]], values)


class TestOutputSplitColumns(unittest.TestCase):
    """Test split_columns_to_files output."""

    def test_split_scalars(self):
        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "split_out")

            @csp.graph
            def g():
                writer = ParquetWriter(
                    outdir,
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    split_columns_to_files=True,
                )
                writer.publish("a", csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(5)]))
                writer.publish("b", csp.curve(float, [(timedelta(seconds=i + 1), i * 2.0) for i in range(5)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=10))
            self.assertTrue(os.path.isdir(outdir))
            self.assertTrue(os.path.isfile(os.path.join(outdir, "a.parquet")))
            self.assertTrue(os.path.isfile(os.path.join(outdir, "b.parquet")))
            self.assertTrue(os.path.isfile(os.path.join(outdir, "csp_timestamp.parquet")))

            # Read back and verify
            df_a = pandas.read_parquet(os.path.join(outdir, "a.parquet"))
            df_b = pandas.read_parquet(os.path.join(outdir, "b.parquet"))
            self.assertEqual(df_a["a"].tolist(), list(range(5)))
            self.assertEqual(df_b["b"].tolist(), [0.0, 2.0, 4.0, 6.0, 8.0])

    def test_split_round_trip_via_csp(self):
        """Write split columns and read back via csp ParquetReader."""
        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "split_data")

            @csp.graph
            def g_write():
                writer = ParquetWriter(
                    outdir,
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    split_columns_to_files=True,
                )
                writer.publish("x", csp.curve(int, [(timedelta(seconds=i + 1), i * 10) for i in range(3)]))

            @csp.graph
            def g_read():
                reader = ParquetReader(outdir, time_column="csp_timestamp", split_columns_to_files=True)
                csp.add_graph_output("x", reader.subscribe_all(int, "x"))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=10))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=10))
            self.assertEqual([v[1] for v in res["x"]], [0, 10, 20])

    def test_split_struct(self):
        class S(csp.Struct):
            a: int
            b: float

        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "split_struct")

            @csp.graph
            def g_write():
                writer = ParquetWriter(
                    outdir,
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    split_columns_to_files=True,
                )
                writer.publish_struct(
                    csp.curve(
                        S,
                        [
                            (timedelta(seconds=1), S(a=1, b=1.1)),
                            (timedelta(seconds=2), S(a=2, b=2.2)),
                        ],
                    )
                )

            @csp.graph
            def g_read():
                reader = ParquetReader(outdir, time_column="csp_timestamp", split_columns_to_files=True)
                csp.add_graph_output("data", reader.subscribe_all(S))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=5))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=5))
            vals = [v[1] for v in res["data"]]
            self.assertEqual(vals[0], S(a=1, b=1.1))
            self.assertEqual(vals[1], S(a=2, b=2.2))

    def test_split_ipc_format(self):
        """split_columns_to_files with write_arrow_binary=True produces .arrow files."""
        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "split_ipc")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, write_arrow_binary=True, compression="")
                writer = ParquetWriter(outdir, "csp_timestamp", config=config, split_columns_to_files=True)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 42)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            self.assertTrue(os.path.isfile(os.path.join(outdir, "x.arrow")))
            self.assertTrue(os.path.isfile(os.path.join(outdir, "csp_timestamp.arrow")))


class TestOutputDictBasket(unittest.TestCase):
    """Test dict basket output."""

    def test_single_file_dict_basket_raises_clear_error(self):
        """Publishing a dict basket on a single-file (non-split) writer fails fast with a clear error."""
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g():
                writer = ParquetWriter(
                    os.path.join(d, "out.parquet"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    # split_columns_to_files defaults to False
                )
                basket = {"AAPL": csp.curve(float, [(timedelta(seconds=1), 1.0)])}
                writer.publish_dict_basket("price", basket, str, float)

            with self.assertRaisesRegex(ValueError, "split_columns_to_files"):
                csp.run(g, starttime=START, endtime=timedelta(seconds=3))

    def test_basic_dict_basket(self):
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g_write():
                writer = ParquetWriter(
                    os.path.join(d, "basket"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    split_columns_to_files=True,
                )
                basket = {
                    "AAPL": csp.curve(float, [(timedelta(seconds=1), 150.0), (timedelta(seconds=2), 151.0)]),
                    "IBM": csp.curve(float, [(timedelta(seconds=1), 130.0), (timedelta(seconds=3), 131.0)]),
                }
                writer.publish_dict_basket("price", basket, str, float)

            @csp.graph
            def g_read():
                reader = ParquetReader(
                    os.path.join(d, "basket"),
                    time_column="csp_timestamp",
                    split_columns_to_files=True,
                )
                basket = reader.subscribe_dict_basket(float, "price", ["AAPL", "IBM"])
                csp.add_graph_output("AAPL", basket["AAPL"])
                csp.add_graph_output("IBM", basket["IBM"])

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=5))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=5))
            aapl = [v[1] for v in res["AAPL"]]
            ibm = [v[1] for v in res["IBM"]]
            self.assertEqual(aapl, [150.0, 151.0])
            self.assertEqual(ibm, [130.0, 131.0])

    def test_dict_basket_many_symbols(self):
        n_symbols = 20
        symbols = [f"SYM{i}" for i in range(n_symbols)]

        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g_write():
                writer = ParquetWriter(
                    os.path.join(d, "basket"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    split_columns_to_files=True,
                )
                basket = {sym: csp.curve(float, [(timedelta(seconds=1), float(i))]) for i, sym in enumerate(symbols)}
                writer.publish_dict_basket("val", basket, str, float)

            @csp.graph
            def g_read():
                reader = ParquetReader(
                    os.path.join(d, "basket"),
                    time_column="csp_timestamp",
                    split_columns_to_files=True,
                )
                basket = reader.subscribe_dict_basket(float, "val", symbols)
                for sym in symbols:
                    csp.add_graph_output(sym, basket[sym])

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=5))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=5))
            for i, sym in enumerate(symbols):
                self.assertEqual([v[1] for v in res[sym]], [float(i)])


class TestOutputNumpyArrays(unittest.TestCase):
    """Test numpy array column output."""

    def test_1d_float_array(self):
        arrays = [numpy.array([1.0, 2.0, 3.0]), numpy.array([4.0, 5.0, 6.0])]

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish(
                    "arr",
                    csp.curve(
                        csp.typing.Numpy1DArray[float],
                        [(timedelta(seconds=i + 1), a) for i, a in enumerate(arrays)],
                    ),
                )

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("arr", reader.subscribe_all(csp.typing.Numpy1DArray[float], "arr"))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=5))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=5))
            for i, (_, arr) in enumerate(res["arr"]):
                numpy.testing.assert_array_equal(arr, arrays[i])

    def test_1d_int_array(self):
        arrays = [numpy.array([10, 20, 30], dtype="int64"), numpy.array([40, 50], dtype="int64")]

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish(
                    "arr",
                    csp.curve(
                        csp.typing.Numpy1DArray[int],
                        [(timedelta(seconds=i + 1), a) for i, a in enumerate(arrays)],
                    ),
                )

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("arr", reader.subscribe_all(csp.typing.Numpy1DArray[int], "arr"))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=5))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=5))
            for i, (_, arr) in enumerate(res["arr"]):
                numpy.testing.assert_array_equal(arr, arrays[i])

    def test_2d_ndarray(self):
        class NDStruct(csp.Struct):
            arr: csp.typing.NumpyNDArray[float]

        arr_2d = numpy.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish_struct(csp.const(NDStruct(arr=arr_2d)))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("data", reader.subscribe_all(NDStruct))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=2))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=2))
            result_arr = res["data"][0][1].arr
            self.assertEqual(result_arr.shape, arr_2d.shape)
            numpy.testing.assert_array_equal(result_arr, arr_2d)

    def test_3d_ndarray(self):
        class NDStruct(csp.Struct):
            arr: csp.typing.NumpyNDArray[float]

        arr_3d = numpy.arange(24.0).reshape((2, 3, 4))

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish_struct(csp.const(NDStruct(arr=arr_3d)))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("data", reader.subscribe_all(NDStruct))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=2))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=2))
            result_arr = res["data"][0][1].arr
            self.assertEqual(result_arr.shape, arr_3d.shape)
            numpy.testing.assert_array_equal(result_arr, arr_3d)


class TestOutputCompression(unittest.TestCase):
    """Test various compression codecs — verify codec is actually applied."""

    def _write_and_check_codec(self, compression, expected_codec, n_rows=100):
        """Write with compression and verify the codec in file metadata."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, compression=compression)
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(n_rows)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=n_rows + 5))
            # Read footer eagerly (no lingering file handle: Windows can't delete an open file)
            actual_codec = pyarrow.parquet.read_metadata(fname).row_group(0).column(0).compression
            self.assertEqual(actual_codec, expected_codec)
            # Verify data integrity
            df = pandas.read_parquet(fname)
            self.assertEqual(df["x"].tolist(), list(range(n_rows)))

    def test_snappy(self):
        self._write_and_check_codec("snappy", "SNAPPY")

    def test_gzip(self):
        self._write_and_check_codec("gzip", "GZIP")

    def test_zstd(self):
        self._write_and_check_codec("zstd", "ZSTD")

    def test_none(self):
        self._write_and_check_codec("", "UNCOMPRESSED")

    def test_compression_name_case_insensitive(self):
        """Upper/mixed-case compression names resolve via Arrow (case-insensitive)."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, compression="ZSTD")
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(10)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=15))
            codec = pyarrow.parquet.read_metadata(fname).row_group(0).column(0).compression
            self.assertEqual(codec, "ZSTD")

    def test_invalid_compression_raises_clear_error(self):
        """An unknown compression name fails with a clear error (not a cryptic one)."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, compression="not_a_codec")
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 1)]))

            with self.assertRaisesRegex(Exception, "compression"):
                csp.run(g, starttime=START, endtime=timedelta(seconds=3))


class TestOutputBatchSize(unittest.TestCase):
    """Test batch_size controls row group flushing."""

    def _write_and_check_row_groups(self, batch_size, n_rows):
        """Write and verify row group count matches batch_size."""
        import math

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, batch_size=batch_size)
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(n_rows)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=n_rows + 5))
            expected_row_groups = math.ceil(n_rows / batch_size)
            self.assertEqual(pyarrow.parquet.read_metadata(fname).num_row_groups, expected_row_groups)
            # Verify data integrity
            df = pandas.read_parquet(fname)
            self.assertEqual(df["x"].tolist(), list(range(n_rows)))

    def test_tiny_batch(self):
        self._write_and_check_row_groups(1, 10)

    def test_small_batch(self):
        self._write_and_check_row_groups(4, 20)

    def test_batch_larger_than_data(self):
        self._write_and_check_row_groups(1000, 10)

    def test_batch_exactly_divides_data(self):
        self._write_and_check_row_groups(5, 15)

    def test_partial_final_batch(self):
        """n_rows not evenly divisible → final partial batch flushed at stop()."""
        self._write_and_check_row_groups(7, 20)  # 20/7 = 2 full + 1 partial = 3 row groups


class TestOutputMetadata(unittest.TestCase):
    """Test file and column metadata."""

    def test_file_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(
                    fname,
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    file_metadata={"created_by": "test", "version": "1.0"},
                )
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 1)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            metadata = pyarrow.parquet.read_schema(fname).metadata
            self.assertEqual(metadata[b"created_by"], b"test")
            self.assertEqual(metadata[b"version"], b"1.0")

    def test_column_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(
                    fname,
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    column_metadata={"x": {"units": "meters", "source": "sensor_1"}},
                )
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 1)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            x_field = pyarrow.parquet.read_schema(fname).field("x")
            self.assertEqual(x_field.metadata[b"units"], b"meters")
            self.assertEqual(x_field.metadata[b"source"], b"sensor_1")

    def test_file_metadata_in_split_mode(self):
        """file_metadata is preserved in split-column mode (regression test for metadata propagation)."""
        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "split_meta")

            @csp.graph
            def g():
                writer = ParquetWriter(
                    outdir,
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    split_columns_to_files=True,
                    file_metadata={"author": "test_suite", "version": "2.0"},
                    column_metadata={"x": {"units": "kg"}},
                )
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 42)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))

            # Each per-column file should carry file-level metadata
            x_file = os.path.join(outdir, "x.parquet")
            ts_file = os.path.join(outdir, "csp_timestamp.parquet")
            self.assertTrue(os.path.isfile(x_file))
            self.assertTrue(os.path.isfile(ts_file))

            schema_x = pyarrow.parquet.read_schema(x_file)
            schema_ts = pyarrow.parquet.read_schema(ts_file)

            # File-level metadata on both files
            self.assertEqual(schema_x.metadata[b"author"], b"test_suite")
            self.assertEqual(schema_x.metadata[b"version"], b"2.0")
            self.assertEqual(schema_ts.metadata[b"author"], b"test_suite")
            self.assertEqual(schema_ts.metadata[b"version"], b"2.0")

            # Column-level metadata preserved on x
            x_field = schema_x.field("x")
            self.assertEqual(x_field.metadata[b"units"], b"kg")

    def test_file_metadata_in_split_ipc_mode(self):
        """file_metadata preserved in split-column + IPC mode."""
        with tempfile.TemporaryDirectory() as d:
            outdir = os.path.join(d, "split_ipc_meta")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, write_arrow_binary=True, compression="")
                writer = ParquetWriter(
                    outdir,
                    "csp_timestamp",
                    config=config,
                    split_columns_to_files=True,
                    file_metadata={"source": "ipc_test"},
                )
                writer.publish("val", csp.curve(int, [(timedelta(seconds=1), 99)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))

            val_file = os.path.join(outdir, "val.arrow")
            self.assertTrue(os.path.isfile(val_file))
            reader = _read_ipc(val_file)
            self.assertEqual(reader.schema.metadata[b"source"], b"ipc_test")


class TestOutputAllowOverwrite(unittest.TestCase):
    """Test allow_overwrite behavior."""

    def test_overwrite_true(self):
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")
            # Write first file
            with open(fname, "w") as f:
                f.write("dummy")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True)
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 42)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertEqual(df["x"].tolist(), [42])

    def test_overwrite_false_raises(self):
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")
            with open(fname, "w") as f:
                f.write("dummy")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=False)
                writer = ParquetWriter(fname, "csp_timestamp", config=config)
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 42)]))

            with self.assertRaises(FileExistsError):
                csp.run(g, starttime=START, endtime=timedelta(seconds=5))


class TestOutputEdgeCases(unittest.TestCase):
    """Edge cases and special scenarios."""

    def test_bare_relative_filename_no_directory(self):
        """Writing to a bare relative filename (no directory component) must work.

        Regression: the C++ sink called mkdir(dirname(path)); dirname("out.parquet")
        is "" and mkdir("") fails with 'Invalid argument'.
        """
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:

                @csp.graph
                def g():
                    writer = ParquetWriter(
                        "out.parquet", "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True)
                    )
                    writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 7)]))

                csp.run(g, starttime=START, endtime=timedelta(seconds=3))
                df = pandas.read_parquet(os.path.join(d, "out.parquet"))
                self.assertEqual(df["x"].tolist(), [7])
            finally:
                os.chdir(cwd)

    def test_bare_relative_directory_split_columns(self):
        """split_columns_to_files with a bare relative directory name must work."""
        with tempfile.TemporaryDirectory() as d:
            cwd = os.getcwd()
            os.chdir(d)
            try:

                @csp.graph
                def g():
                    writer = ParquetWriter(
                        "split_out",
                        "csp_timestamp",
                        config=ParquetOutputConfig(allow_overwrite=True),
                        split_columns_to_files=True,
                    )
                    writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 7)]))

                csp.run(g, starttime=START, endtime=timedelta(seconds=3))
                df = pandas.read_parquet(os.path.join(d, "split_out", "x.parquet"))
                self.assertEqual(df["x"].tolist(), [7])
            finally:
                os.chdir(cwd)

    def test_multiple_ticks_same_timestamp(self):
        """Multiple values at the same engine time produce multiple rows."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                # csp.curve with same timedelta produces ticks at same engine time
                writer.publish(
                    "x",
                    csp.curve(
                        int, [(timedelta(seconds=1), 10), (timedelta(seconds=1), 20), (timedelta(seconds=1), 30)]
                    ),
                )

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertEqual(df["x"].tolist(), [10, 20, 30])

    def test_empty_output_no_ticks(self):
        """No data ticked — file should still be created with schema."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                # Curve with data beyond endtime → no ticks
                writer.publish("x", csp.curve(int, [(timedelta(seconds=100), 1)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertEqual(len(df), 0)
            self.assertIn("x", df.columns)

    def test_large_number_of_columns(self):
        """Write 100 columns."""
        n_cols = 100
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                for c in range(n_cols):
                    writer.publish(f"col_{c}", csp.curve(int, [(timedelta(seconds=1), c)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertEqual(len(df.columns), n_cols + 1)  # +1 for timestamp
            for c in range(n_cols):
                self.assertEqual(df[f"col_{c}"].iloc[0], c)

    def test_creates_parent_directories(self):
        """Writer creates parent directories if they don't exist."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "sub", "dir", "deep", "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 1)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            self.assertTrue(os.path.isfile(fname))

    def test_duplicate_column_name_raises(self):
        """Publishing same column name twice raises."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 1)]))
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 2)]))

            with self.assertRaises(KeyError):
                csp.run(g, starttime=START, endtime=timedelta(seconds=5))

    def test_very_long_strings(self):
        """Write very long string values."""
        long_str = "x" * 100_000
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish("s", csp.curve(str, [(timedelta(seconds=1), long_str)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertEqual(df["s"].iloc[0], long_str)

    def test_write_read_round_trip_all_types(self):
        """Full round-trip test with all supported types in one file."""

        class MyEnum(csp.Enum):
            X = 1
            Y = 2

        class FullStruct(csp.Struct):
            i: int
            f: float
            b: bool
            s: str
            dt: datetime
            td: timedelta
            d: date
            t: time
            e: MyEnum

        v = FullStruct(
            i=42,
            f=3.14,
            b=True,
            s="hello",
            dt=datetime(2022, 6, 15, 12, 0, tzinfo=pytz.utc),
            td=timedelta(seconds=123),
            d=date(2022, 6, 15),
            t=time(12, 30, 45),
            e=MyEnum.X,
        )

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g_write():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish_struct(csp.const(v))

            @csp.graph
            def g_read():
                reader = ParquetReader(fname, time_column="csp_timestamp")
                csp.add_graph_output("data", reader.subscribe_all(FullStruct))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=2))
            res = csp.run(g_read, starttime=START, endtime=START + timedelta(seconds=2))
            read_v = res["data"][0][1]
            self.assertEqual(read_v, v)

    def test_float_nan_round_trip(self):
        """NaN values round-trip correctly."""
        import math

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g():
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish(
                    "x",
                    csp.curve(
                        float,
                        [
                            (timedelta(seconds=1), 1.0),
                            (timedelta(seconds=2), float("nan")),
                            (timedelta(seconds=3), 3.0),
                        ],
                    ),
                )

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            df = pandas.read_parquet(fname)
            self.assertEqual(df["x"].iloc[0], 1.0)
            self.assertTrue(math.isnan(df["x"].iloc[1]))
            self.assertEqual(df["x"].iloc[2], 3.0)

    def test_struct_with_unset_fields_produces_nulls(self):
        """Struct fields that are never set produce null cells."""

        class Partial(csp.Struct):
            a: int
            b: float
            c: str

        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.parquet")

            @csp.graph
            def g_write():
                # Only set 'a', leave 'b' and 'c' unset
                v = Partial(a=42)
                writer = ParquetWriter(fname, "csp_timestamp", config=ParquetOutputConfig(allow_overwrite=True))
                writer.publish_struct(csp.const(v))

            csp.run(g_write, starttime=START, endtime=timedelta(seconds=2))
            df = pandas.read_parquet(fname)
            self.assertEqual(df["a"].iloc[0], 42)
            self.assertTrue(pandas.isna(df["b"].iloc[0]))
            self.assertTrue(pandas.isna(df["c"].iloc[0]))

    def test_multi_batch_with_rotation(self):
        """Rows spanning multiple batches + file rotation land in correct files."""
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g():
                # 20 rows, batch_size=4, rotate at row 10
                data = csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(20)])
                rotate = csp.curve(str, [(timedelta(seconds=11), os.path.join(d, "p1.parquet"))])
                config = ParquetOutputConfig(allow_overwrite=True, batch_size=4)
                writer = ParquetWriter(
                    os.path.join(d, "p0.parquet"),
                    "csp_timestamp",
                    config=config,
                    filename_provider=rotate,
                )
                writer.publish("value", data)

            csp.run(g, starttime=START, endtime=timedelta(seconds=25))
            df0 = pandas.read_parquet(os.path.join(d, "p0.parquet"))
            df1 = pandas.read_parquet(os.path.join(d, "p1.parquet"))
            # First 10 rows in p0, next 10 in p1
            self.assertEqual(df0["value"].tolist(), list(range(10)))
            self.assertEqual(df1["value"].tolist(), list(range(10, 20)))
            # Verify batch_size=4 → correct row group counts
            import math

            self.assertEqual(
                pyarrow.parquet.read_metadata(os.path.join(d, "p0.parquet")).num_row_groups, math.ceil(10 / 4)
            )  # 3
            self.assertEqual(
                pyarrow.parquet.read_metadata(os.path.join(d, "p1.parquet")).num_row_groups, math.ceil(10 / 4)
            )  # 3

    def test_file_visitor_exact_contract(self):
        """file_visitor called once per closed file, in order, including final at stop()."""
        visited = []
        with tempfile.TemporaryDirectory() as d:

            @csp.graph
            def g():
                data = csp.curve(int, [(timedelta(seconds=i + 1), i) for i in range(9)])
                rotate = csp.curve(
                    str,
                    [
                        (timedelta(seconds=4), os.path.join(d, "p1.parquet")),
                        (timedelta(seconds=7), os.path.join(d, "p2.parquet")),
                    ],
                )
                writer = ParquetWriter(
                    os.path.join(d, "p0.parquet"),
                    "csp_timestamp",
                    config=ParquetOutputConfig(allow_overwrite=True),
                    filename_provider=rotate,
                    file_visitor=lambda f: visited.append(f),
                )
                writer.publish("value", data)

            csp.run(g, starttime=START, endtime=timedelta(seconds=15))
            # Expect 3 visits: p0 (closed at rotation 1), p1 (closed at rotation 2), p2 (closed at stop)
            self.assertEqual(len(visited), 3)
            self.assertTrue(visited[0].endswith("p0.parquet"))
            self.assertTrue(visited[1].endswith("p1.parquet"))
            self.assertTrue(visited[2].endswith("p2.parquet"))
            # Each visited file should be readable
            for f in visited:
                pandas.read_parquet(f)

    def test_metadata_in_ipc_mode(self):
        """file_metadata and column_metadata work in Arrow IPC mode."""
        with tempfile.TemporaryDirectory() as d:
            fname = os.path.join(d, "out.arrow")

            @csp.graph
            def g():
                config = ParquetOutputConfig(allow_overwrite=True, write_arrow_binary=True, compression="")
                writer = ParquetWriter(
                    fname,
                    "csp_timestamp",
                    config=config,
                    file_metadata={"author": "test"},
                    column_metadata={"x": {"units": "kg"}},
                )
                writer.publish("x", csp.curve(int, [(timedelta(seconds=1), 1)]))

            csp.run(g, starttime=START, endtime=timedelta(seconds=5))
            reader = _read_ipc(fname)
            schema = reader.schema
            self.assertEqual(schema.metadata[b"author"], b"test")
            x_field = schema.field("x")
            self.assertEqual(x_field.metadata[b"units"], b"kg")


if __name__ == "__main__":
    unittest.main()
