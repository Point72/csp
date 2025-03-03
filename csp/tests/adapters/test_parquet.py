import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy
import pandas
import polars
import pyarrow
import pyarrow.parquet
import pytz

import csp
from csp.adapters.output_adapters.parquet import ParquetOutputConfig
from csp.adapters.parquet import ParquetReader, ParquetWriter


class PriceQuantity(csp.Struct):
    PRICE: float
    SIZE: int
    SIDE: str
    SYMBOL: str


class PriceQuantity2(csp.Struct):
    price: float
    quantity: int
    side: str


parquet_filename = os.path.join(os.path.dirname(__file__), "parquet_test_data.parquet")
arrow_filename = os.path.join(os.path.dirname(__file__), "arrow_test_data.arrow")


class TestParquet(unittest.TestCase):
    def do_test_body(self, filename, arrow=False):
        temp_file = tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w")
        temp_file.close()

        @csp.graph
        def write_ts_to_file(x: csp.ts["T"]):
            if arrow:
                config = ParquetOutputConfig(write_arrow_binary=True)
            else:
                config = ParquetOutputConfig()
            parquet_writer = ParquetWriter(
                file_name=temp_file.name,
                timestamp_column_name="TIME",
                config=config,
            )
            parquet_writer.publish_struct(x)

        def graph():
            if arrow:
                reader = ParquetReader(
                    filename, symbol_column="SYMBOL", time_column="TIME", binary_arrow=arrow, tz=pytz.utc
                )
                reader_all = ParquetReader(
                    filename, symbol_column="SYMBOL", time_column="TIME", binary_arrow=arrow, tz=pytz.utc
                )
            else:
                reader = ParquetReader(filename, symbol_column="SYMBOL", time_column="TIME", tz=pytz.utc)
                reader_all = ParquetReader(filename, symbol_column="SYMBOL", time_column="TIME", tz=pytz.utc)

            # Struct
            aapl = reader.subscribe("AAPL", PriceQuantity)
            ibm = reader.subscribe("IBM", PriceQuantity)

            # Struct with fieldMapping
            aapl2 = reader.subscribe(
                "AAPL", PriceQuantity2, field_map={"PRICE": "price", "SIZE": "quantity", "SIDE": "side"}
            )

            # specific field
            aapl_price = reader.subscribe("AAPL", float, field_map="PRICE")

            # all data
            all = reader_all.subscribe_all(PriceQuantity)
            write_ts_to_file(all)

            csp.add_graph_output("aapl", aapl)
            csp.add_graph_output("ibm", ibm)
            csp.add_graph_output("aapl2", aapl2)
            csp.add_graph_output("aapl_price", aapl_price)
            csp.add_graph_output("all", all)

        result = csp.run(graph, starttime=datetime(2020, 3, 3, 9, 30))
        os.unlink(temp_file.name)
        self.assertEqual(len(result["aapl"]), 4)
        self.assertTrue(all(v[1].SYMBOL == "AAPL" for v in result["aapl"]))

        self.assertEqual(len(result["ibm"]), 2)
        self.assertTrue(all(v[1].SYMBOL == "IBM" for v in result["ibm"]))

        self.assertEqual(
            [v[1] for v in result["aapl"]],
            [
                PriceQuantity(PRICE=500.0, SIZE=100, SIDE="BUY", SYMBOL="AAPL"),
                PriceQuantity(PRICE=400.0, SIZE=100, SIDE="BUY", SYMBOL="AAPL"),
                PriceQuantity(PRICE=300.0, SIZE=200, SIDE="SELL", SYMBOL="AAPL"),
                PriceQuantity(PRICE=200.0, SIZE=400, SIDE="BUY", SYMBOL="AAPL"),
            ],
        )

        self.assertEqual(
            [v[1] for v in result["aapl2"]],
            [
                PriceQuantity2(price=500.0, quantity=100, side="BUY"),
                PriceQuantity2(price=400.0, quantity=100, side="BUY"),
                PriceQuantity2(
                    price=300.0,
                    quantity=200,
                    side="SELL",
                ),
                PriceQuantity2(price=200.0, quantity=400, side="BUY"),
            ],
        )

        self.assertEqual([v[1] for v in result["aapl_price"]], [500.0, 400.0, 300.0, 200.0])
        self.assertEqual(len(result["all"]), 7)

    def test_arrow(self):
        self.do_test_body(arrow_filename, True)

    def test_parquet(self):
        self.do_test_body(parquet_filename)

    def test_parquet_output_file_rotation(self):
        seen_filenames = []

        def dummy_file_visitor(x: str):
            seen_filenames.append(x)
            self.assertTrue(os.path.isfile(x))
            # tests that we can actually open the parquet file
            pandas.read_parquet(x)

        @csp.graph
        def g(d: str, empty_last_file: bool = True):
            values = [(timedelta(seconds=i), i) for i in range(10)]
            if not empty_last_file:
                values += [(timedelta(seconds=11), 11)]
            data = csp.curve(int, values)
            rotate_file = csp.curve(
                str,
                [
                    (timedelta(seconds=4), os.path.join(d, "out1.parquet")),
                    (timedelta(seconds=4.5), os.path.join(d, "out2.parquet")),
                    (timedelta(seconds=4.6), os.path.join(d, "out3.parquet")),
                    (timedelta(seconds=11), os.path.join(d, "out4.parquet")),
                ],
            )

            parquet_writer = ParquetWriter(
                file_name=os.path.expanduser(os.path.join(d, "out0.parquet")),
                timestamp_column_name="csp_timestamp",
                filename_provider=rotate_file,
                file_visitor=dummy_file_visitor,
            )
            parquet_writer.publish("my_out", data)

        with tempfile.TemporaryDirectory() as d:
            csp.run(g, d, starttime=datetime.now(), endtime=timedelta(seconds=20))
            dfs = [pandas.read_parquet(os.path.join(d, f"out{i}.parquet")) for i in range(5)]
            self.assertEqual(dfs[0].my_out.tolist(), list(range(4)))
            self.assertEqual(dfs[1].my_out.tolist(), [4])
            self.assertEqual(dfs[2].my_out.tolist(), [])
            self.assertEqual(dfs[3].my_out.tolist(), list(range(5, 10)))
            self.assertEqual(dfs[4].my_out.tolist(), [])
            self.assertEqual(seen_filenames, [str(os.path.join(d, f"out{i}.parquet")) for i in range(5)])
        seen_filenames = []

        with tempfile.TemporaryDirectory() as d:
            csp.run(g, d, False, starttime=datetime.now(), endtime=timedelta(seconds=20))
            dfs = [pandas.read_parquet(os.path.join(d, f"out{i}.parquet")) for i in range(5)]
            self.assertEqual(dfs[0].my_out.tolist(), list(range(4)))
            self.assertEqual(dfs[1].my_out.tolist(), [4])
            self.assertEqual(dfs[2].my_out.tolist(), [])
            self.assertEqual(dfs[3].my_out.tolist(), list(range(5, 10)))
            self.assertEqual(dfs[4].my_out.tolist(), [11])
            self.assertEqual(seen_filenames, [str(os.path.join(d, f"out{i}.parquet")) for i in range(5)])

    def test_parquet_writer(self):
        with tempfile.TemporaryDirectory() as d:
            # removing to ensure adapter creates the directory
            os.rmdir(d)

            for timestamp_column_name in ["timestamp", None]:
                filename = os.path.join(d, "test.parquet")

                @csp.graph
                def graph():
                    x = csp.curve(int, [(timedelta(seconds=v + 1), v + 1) for v in range(10)])
                    y = csp.curve(float, [(timedelta(seconds=v + 1), v * 10.0) for v in range(10)])
                    writer = ParquetWriter(
                        file_name=filename,
                        timestamp_column_name=timestamp_column_name,
                        config=ParquetOutputConfig(allow_overwrite=True),
                    )
                    writer.publish("x", x)
                    writer.publish("y", y)

                start_time = datetime(2020, 3, 3, 9, 30, tzinfo=pytz.utc)
                csp.run(graph, starttime=start_time)
                df = pandas.read_parquet(filename)
                expected_columns = {}
                if timestamp_column_name:
                    expected_columns[timestamp_column_name] = [
                        start_time + timedelta(seconds=sec + 1) for sec in range(10)
                    ]
                expected_columns.update(
                    {
                        "x": list(range(1, 11)),
                        "y": list(map(float, range(0, 100, 10))),
                    }
                )

                expected_df = pandas.DataFrame.from_dict(expected_columns)
                self.assertTrue((df.dtypes == expected_df.dtypes).all())
                self.assertTrue((expected_df == df).all().all())

    def test_arrow_writer(self):
        with tempfile.TemporaryDirectory() as d:
            os.rmdir(d)
            filename = os.path.join(d, "test.arrow")

            @csp.graph
            def graph():
                x = csp.curve(int, [(timedelta(seconds=v + 1), v + 1) for v in range(10)])
                y = csp.curve(float, [(timedelta(seconds=v + 1), v * 10.0) for v in range(10)])
                config = ParquetOutputConfig(allow_overwrite=True, write_arrow_binary=True, compression="")
                writer = ParquetWriter(file_name=filename, timestamp_column_name="timestamp", config=config)
                writer.publish("x", x)
                writer.publish("y", y)

            start_time = datetime(2020, 3, 3, 9, 30, tzinfo=pytz.utc)
            csp.run(graph, starttime=start_time)

            df = pyarrow.RecordBatchStreamReader(pyarrow.memory_map(filename)).read_pandas()

            expected_df = pandas.DataFrame.from_dict(
                {
                    "timestamp": [start_time + timedelta(seconds=sec + 1) for sec in range(10)],
                    "x": list(range(1, 11)),
                    "y": list(map(float, range(0, 100, 10))),
                }
            )
            self.assertTrue((df.dtypes == expected_df.dtypes).all())
            self.assertTrue((expected_df == df).all().all())

    def test_parquet_multiple_values_same_tick(self):
        @csp.graph
        def writer_g(output_file: str):
            symbol = csp.curve(str, [(timedelta(0), "AAPL"), (timedelta(0), "IBM"), (timedelta(0), "Z")])
            value = csp.curve(float, [(timedelta(0), 100.0), (timedelta(0), 200.0), (timedelta(0), 300.0)])
            parquet_writer = ParquetWriter(output_file, "csp_timestamp")
            parquet_writer.publish("symbol", symbol)
            parquet_writer.publish("value", value)
            csp.add_graph_output("symbol", symbol)
            csp.add_graph_output("value", value)

        @csp.graph
        def reader_g1(input_file: str):
            reader = ParquetReader(input_file, "symbol", "csp_timestamp")
            csp.add_graph_output("symbol", reader.subscribe_all(str, "symbol"))
            csp.add_graph_output("value", reader.subscribe_all(float, "value"))

        @csp.graph
        def reader_g2(input_file: str):
            reader = ParquetReader(input_file, "symbol", "csp_timestamp")
            reader_all = ParquetReader(input_file, "symbol", "csp_timestamp")
            csp.add_graph_output(
                "symbols",
                csp.collect(
                    [
                        reader.subscribe("AAPL", str, "symbol"),
                        reader.subscribe("IBM", str, "symbol"),
                        reader.subscribe("Z", str, "symbol"),
                    ]
                ),
            )
            csp.add_graph_output(
                "values",
                csp.collect(
                    [
                        reader.subscribe("AAPL", float, "value"),
                        reader.subscribe("IBM", float, "value"),
                        reader.subscribe("Z", float, "value"),
                    ]
                ),
            )
            csp.add_graph_output("all_symbols", reader_all.subscribe_all(str, "symbol"))
            csp.add_graph_output("all_values", reader_all.subscribe_all(float, "value"))

        @csp.graph
        def reader_g3(input_file: str):
            reader = ParquetReader(input_file, "symbol", "csp_timestamp")
            csp.add_graph_output(
                "symbols",
                csp.collect(
                    [
                        reader.subscribe("AAPL", str, "symbol"),
                        reader.subscribe("IBM", str, "symbol"),
                        reader.subscribe("Z", str, "symbol"),
                    ]
                ),
            )
            csp.add_graph_output(
                "values",
                csp.collect(
                    [
                        reader.subscribe("AAPL", float, "value"),
                        reader.subscribe("IBM", float, "value"),
                        reader.subscribe("Z", float, "value"),
                    ]
                ),
            )

        with tempfile.TemporaryDirectory(prefix="csp_unit_tests") as tmp_folder:
            file_name = os.path.join(tmp_folder, "data.parquet")
            ref_data = csp.run(writer_g, file_name, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))
            read_data1 = csp.run(reader_g1, file_name, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))
            self.assertEqual(ref_data, read_data1)
            read_data2 = csp.run(reader_g2, file_name, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))
            read_data3 = csp.run(reader_g3, file_name, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))

            # When we have subscribe all we want all to tick on separate cycles
            self.assertEqual(len(read_data2["symbols"]), 1)
            self.assertEqual(len(read_data2["values"]), 1)
            self.assertEqual(len(read_data2["all_symbols"]), 3)
            self.assertEqual(len(read_data2["all_values"]), 3)

            # When we have don't have subscribe all, we want all to tick on the same cycle
            self.assertEqual(len(read_data3["symbols"]), 1)
            self.assertEqual(len(read_data3["values"]), 1)
            self.assertEqual(sorted(read_data3["symbols"][0][1]), ["AAPL", "IBM", "Z"])

    def test_duplicate_field_name(self):
        """better errors in parquet writer"""

        class MyStruct(csp.Struct):
            time: datetime
            value: int

        def writer_graph(file_name, arrow: bool = False):
            config = ParquetOutputConfig(write_arrow_binary=True) if arrow else None
            parquet_writer = ParquetWriter(file_name=file_name, timestamp_column_name="time", config=config)
            values = csp.curve(
                MyStruct,
                [
                    (timedelta(seconds=1), MyStruct(time=datetime.now(), value=1)),
                    (timedelta(seconds=2), MyStruct(time=datetime.now(), value=2)),
                    (timedelta(seconds=3), MyStruct(time=datetime.now(), value=3)),
                    (timedelta(seconds=4), MyStruct(time=datetime.now(), value=4)),
                ],
            )
            with self.assertRaisesRegex(KeyError, ".*Publishing duplicate column names in parquet/arrow file: time.*"):
                parquet_writer.publish_struct(values)

            # This should work
            parquet_writer.publish_struct(values, field_map={"time": "new_time", "value": "value"})

            with self.assertRaisesRegex(KeyError, ".*Publishing duplicate column names in parquet/arrow file: value.*"):
                parquet_writer.publish("value", csp.null_ts(int))

        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as f:
            f.close()
            csp.run(writer_graph, file_name=f.name, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))
            os.unlink(f.name)

    def test_non_python_c_fields(self):
        # There was a bug when setting struct fields if the type wasn't the same as the python "type" for this variable.
        # For example if a column in the file is int32 then it wouldn't be properly set on int field of a struct
        start_t = datetime(2021, 1, 1)
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as f:
            f.close()
            with open(f.name, "wb") as w:
                schema = pyarrow.schema([("timestamp", pyarrow.timestamp("ns")), ("int32_val", pyarrow.int32())])
                writer = pyarrow.RecordBatchStreamWriter(w, schema)
                writer.write_table(
                    pyarrow.table(
                        [
                            pyarrow.array([start_t + timedelta(seconds=s) for s in range(5)]),
                            pyarrow.array([0, 1, 2, 3, 4], type=pyarrow.int32()),
                        ],
                        schema=schema,
                    )
                )

            for field_type in (int, float):

                class MyStruct(csp.Struct):
                    int32_val: field_type

                @csp.graph
                def g() -> csp.ts[MyStruct]:
                    reader = ParquetReader(f.name, time_column="timestamp", binary_arrow=True)
                    return reader.subscribe_all(MyStruct, MyStruct.default_field_map())

                res = csp.run(g, starttime=start_t, endtime=timedelta(seconds=10))
                self.assertEqual([v.int32_val for k, v in res[0]], [0, 1, 2, 3, 4])

    def test_in_memory_arrow_read(self):
        class MyS(csp.Struct):
            symbol: str
            exch: str
            size: int
            price: float
            qt: timedelta

        s = datetime(2021, 1, 1)
        vals = {
            "timestamp": pyarrow.array([s + timedelta(seconds=i) for i in range(5)]),
            "symbol": pyarrow.array(["AAPL", "AAPL", "IBM", "AAPL", "IBM"]).dictionary_encode(),
            "exch": pyarrow.array(["Q", "N", "Q", "Q", "A"], pyarrow.binary(1)),
            "size": pyarrow.array([1, 2, 3, 4, 5], pyarrow.int32()),
            "price": pyarrow.array([100, 100.5, 200, 99, 200.8]),
            "qt": pyarrow.array([timedelta(seconds=i) for i in range(5)], pyarrow.duration("ns")),
        }
        t = pyarrow.table(list(vals.values()), list(vals.keys()))
        t1 = t.slice(0, 3)
        t2 = t.slice(3)

        @csp.graph(memoize=False)
        def read_tables(tables: object):
            reader = ParquetReader(tables, time_column="timestamp", binary_arrow=True, read_from_memory_tables=True)
            records = reader.subscribe_all(MyS, MyS.default_field_map())
            csp.add_graph_output("o", records)

        def res_to_df(res):
            times, structs = list(zip(*res["o"]))
            d = {"timestamp": times}
            d.update({k: [getattr(s, k) for s in structs] for k in MyS.metadata()})
            d["exch"] = [v.encode() for v in d["exch"]]
            return pandas.DataFrame(d)

        empty_t = t.slice(0, 0)

        # Simple single table test
        resT1 = csp.run(read_tables, t1, starttime=s, endtime=timedelta(seconds=10))
        self.assertTrue((t1.to_pandas() == res_to_df(resT1)).all().all())
        # Single table with multiple chunks of data in it
        resT = csp.run(read_tables, pyarrow.concat_tables([t1, t2]), starttime=s, endtime=timedelta(seconds=10))
        self.assertTrue((t.to_pandas() == res_to_df(resT)).all().all())
        # Single table with multiple chunks of data in it and empty chunk in the middle
        resT = csp.run(
            read_tables,
            pyarrow.concat_tables([empty_t, t1, empty_t, t2, empty_t]),
            starttime=s,
            endtime=timedelta(seconds=10),
        )
        self.assertTrue((t.to_pandas() == res_to_df(resT)).all().all())
        # List of tables
        resT = csp.run(read_tables, [t1, t2], starttime=s, endtime=timedelta(seconds=10))
        self.assertTrue((t.to_pandas() == res_to_df(resT)).all().all())
        # List of tables with empty tables
        resT = csp.run(read_tables, [empty_t, t1, empty_t, t2, empty_t], starttime=s, endtime=timedelta(seconds=10))
        self.assertTrue((t.to_pandas() == res_to_df(resT)).all().all())
        # Generator of tables
        resT = csp.run(read_tables, (v for v in [t1, t2]), starttime=s, endtime=timedelta(seconds=10))
        self.assertTrue((t.to_pandas() == res_to_df(resT)).all().all())
        # A function that's given the start and end time of the run generates the tables
        resT = csp.run(read_tables, lambda s, e: (v for v in [t1, t2]), starttime=s, endtime=timedelta(seconds=10))
        self.assertTrue((t.to_pandas() == res_to_df(resT)).all().all())
        # Empty table
        resEmpty = csp.run(read_tables, empty_t, starttime=s, endtime=timedelta(seconds=10))
        self.assertEqual(len(resEmpty["o"]), 0)
        resEmpty = csp.run(read_tables, [empty_t, empty_t], starttime=s, endtime=timedelta(seconds=10))
        self.assertEqual(len(resEmpty["o"]), 0)
        with self.assertRaisesRegex(TypeError, ".*Expected PyTable from generator, got str.*"):
            csp.run(read_tables, ["dummy"], starttime=s, endtime=timedelta(seconds=10))

    def test_bytes_read_write(self):
        VALUE = b"my" + bytes([0]) + b"_value"

        @csp.graph
        def g_write(temp_file: object):
            writer = ParquetWriter(temp_file, "csp_timestamp")
            writer.publish("value", csp.const(VALUE))

        @csp.node
        def pass_through(x: csp.ts[bytes]) -> csp.ts[bytes]:
            self.assertEqual(x, VALUE)
            return x

        @csp.graph
        def g_read(temp_file: object) -> csp.Outputs(v1=csp.ts[bytes], v2=csp.ts[bytes]):
            reader = ParquetReader(temp_file, time_column="csp_timestamp")
            res = reader.subscribe_all(bytes, "value")
            return csp.output(v1=res, v2=pass_through(res))

        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            csp.set_capture_cpp_backtrace(True)
            temp_file.close()
            s = datetime(2021, 1, 1)
            e = timedelta(seconds=10)
            csp.run(g_write, temp_file.name, starttime=s, endtime=e)
            res = csp.run(g_read, temp_file.name, starttime=s, endtime=e)
            self.assertEqual(res["v1"], [(datetime(2021, 1, 1, 0, 0), VALUE)])
            self.assertEqual(res["v2"], [(datetime(2021, 1, 1, 0, 0), VALUE)])

    def test_parquet_mismatched_column_type_error(self):
        """Tests that we provide the name of the column in the parquet file that has a mismatched type
        :return:
        """

        @csp.graph
        def g_write(temp_file: object):
            writer = ParquetWriter(temp_file, "csp_timestamp")
            writer.publish("symbol", csp.const(123))
            writer.publish("value", csp.const(1))

        @csp.graph
        def g_read(temp_file: object) -> csp.ts[int]:
            reader = ParquetReader(temp_file, symbol_column="symbol", time_column="csp_timestamp")
            res = reader.subscribe("sym", int, "value")
            return res

        @csp.graph
        def g_read2(temp_file: object) -> csp.ts[str]:
            reader = ParquetReader(temp_file, time_column="csp_timestamp")
            res = reader.subscribe_all(str, "value")
            return res

        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            s = datetime(2021, 1, 1)
            e = timedelta(seconds=10)
            csp.run(g_write, temp_file.name, starttime=s, endtime=e)
            with self.assertRaisesRegex(
                TypeError, ".*Provided symbol type does not match symbol column type \(int64\)"
            ):
                csp.run(g_read, temp_file.name, starttime=s, endtime=e)
            with self.assertRaisesRegex(
                TypeError, ".*Unexpected column type for column value , expected STRING got int64.*"
            ):
                csp.run(g_read2, temp_file.name, starttime=s, endtime=e)

    def test_parquet_schema_change(self):
        class SubStruct(csp.Struct):
            value2: int

        class MyStruct(csp.Struct):
            value: int

        class MyStruct2(MyStruct):
            sub: SubStruct

        @csp.graph
        def g_write(temp_file: object, base_value: int, write_struct2: bool = False, write_value2: bool = False):
            writer = ParquetWriter(temp_file, "csp_timestamp")
            writer.publish("symbol", csp.const("sym"))
            if write_struct2:
                sub_struct = SubStruct()
                res_struct = MyStruct2(value=base_value, sub=sub_struct)
                if write_value2:
                    sub_struct.value2 = base_value * 10
                writer.publish_struct(csp.const(res_struct))
            else:
                writer.publish_struct(csp.const(MyStruct(value=base_value)))

        @csp.graph
        def g_read(temp_files: object, read_value2: bool = False, allow_missing_columns: bool = True) -> csp.Outputs(
            value=csp.ts[int], value2=csp.ts[int]
        ):
            reader = ParquetReader(
                temp_files,
                symbol_column="symbol",
                time_column="csp_timestamp",
                allow_missing_columns=allow_missing_columns,
            )
            value = reader.subscribe("sym", int, "value")
            if read_value2:
                value2 = reader.subscribe("sym", SubStruct, "sub").value2
            else:
                value2 = csp.null_ts(int)
            return csp.output(value=value, value2=value2)

        @csp.graph
        def g_read_struct(temp_files: object, struct_type: "T", allow_missing_columns: bool = True) -> csp.ts["T"]:
            reader = ParquetReader(
                temp_files,
                symbol_column="symbol",
                time_column="csp_timestamp",
                allow_missing_columns=allow_missing_columns,
            )
            return reader.subscribe("sym", struct_type)

        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file1:
            temp_file1.close()
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file2:
                temp_file2.close()
                with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file3:
                    temp_file3.close()
                    with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file4:
                        temp_file4.close()
                        s1 = datetime(2021, 1, 1)
                        e1 = s1 + timedelta(seconds=10, microseconds=-1)
                        s2 = e1 + timedelta(microseconds=1)
                        e2 = s2 + timedelta(seconds=10, microseconds=-1)
                        s3 = e2 + timedelta(microseconds=1)
                        e3 = s3 + timedelta(seconds=10, microseconds=-1)
                        s4 = e3 + timedelta(microseconds=1)
                        e4 = s4 + timedelta(seconds=10, microseconds=-1)
                        csp.run(g_write, temp_file1.name, 1, starttime=s1, endtime=e1)
                        csp.run(g_write, temp_file2.name, 2, True, starttime=s2, endtime=e2)
                        csp.run(g_write, temp_file3.name, 3, True, True, starttime=s3, endtime=e3)
                        csp.run(g_write, temp_file4.name, 4, starttime=s4, endtime=e4)
                        file_names = [temp_file1.name, temp_file2.name, temp_file3.name, temp_file4.name]

                        with self.assertRaisesRegex(RuntimeError, ".*Missing column sub.*"):
                            csp.run(g_read, file_names, True, starttime=s1, endtime=e4, allow_missing_columns=False)

                        with self.assertRaisesRegex(RuntimeError, ".*Missing column sub.*"):
                            csp.run(
                                g_read_struct,
                                file_names,
                                MyStruct2,
                                starttime=s1,
                                endtime=e4,
                                allow_missing_columns=False,
                            )

                        res = csp.run(g_read, file_names, True, starttime=s1, endtime=e4)
                        res2 = csp.run(
                            g_read_struct, file_names, MyStruct, starttime=s1, endtime=e4, allow_missing_columns=False
                        )
                        res3 = csp.run(g_read_struct, file_names, MyStruct2, starttime=s1, endtime=e4)

                        self.assertEqual(
                            res,
                            {
                                "value": [
                                    (datetime(2021, 1, 1, 0, 0), 1),
                                    (datetime(2021, 1, 1, 0, 0, 10), 2),
                                    (datetime(2021, 1, 1, 0, 0, 20), 3),
                                    (datetime(2021, 1, 1, 0, 0, 30), 4),
                                ],
                                "value2": [(datetime(2021, 1, 1, 0, 0, 20), 30)],
                            },
                        )
                        self.assertEqual(
                            res2,
                            {
                                0: [
                                    (datetime(2021, 1, 1, 0, 0), MyStruct(value=1)),
                                    (datetime(2021, 1, 1, 0, 0, 10), MyStruct(value=2)),
                                    (datetime(2021, 1, 1, 0, 0, 20), MyStruct(value=3)),
                                    (datetime(2021, 1, 1, 0, 0, 30), MyStruct(value=4)),
                                ]
                            },
                        )
                        self.assertEqual(
                            res3,
                            {
                                0: [
                                    (datetime(2021, 1, 1, 0, 0), MyStruct2(value=1)),
                                    (datetime(2021, 1, 1, 0, 0, 10), MyStruct2(value=2, sub=SubStruct())),
                                    (datetime(2021, 1, 1, 0, 0, 20), MyStruct2(value=3, sub=SubStruct(value2=30))),
                                    (datetime(2021, 1, 1, 0, 0, 30), MyStruct2(value=4)),
                                ]
                            },
                        )

    def test_parquet_missing_files_read(self):
        @csp.graph
        def g_write(f1: str, f2: str):
            writer1 = ParquetWriter(f1, "timestamp")
            writer2 = ParquetWriter(f2, "timestamp")

            writer1.publish("value", csp.curve(int, [(timedelta(seconds=1), 1), (timedelta(seconds=2), 2)]))
            writer2.publish("value", csp.curve(int, [(timedelta(seconds=3), 3), (timedelta(seconds=4), 4)]))

        @csp.graph
        def g_read(file_names: object, allow_missing_files: bool = False) -> csp.ts[int]:
            if allow_missing_files:
                reader = ParquetReader(file_names, time_column="timestamp", allow_missing_files=allow_missing_files)
            else:
                reader = ParquetReader(file_names, time_column="timestamp")
            return reader.subscribe_all(int, "value")

        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests_f1", mode="w") as temp_file1:
            temp_file1.close()
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests_f2", mode="w") as temp_file2:
                temp_file2.close()
                csp.set_capture_cpp_backtrace(True)
                csp.run(
                    g_write,
                    temp_file1.name,
                    temp_file2.name,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=60),
                )
                with self.assertRaisesRegex(Exception, ".*Failed to open local file"):
                    res = csp.run(
                        g_read,
                        [temp_file1.name, "dummy", temp_file2.name],
                        starttime=datetime(2022, 1, 1),
                        endtime=timedelta(seconds=60),
                    )

                res = csp.run(
                    g_read,
                    [temp_file1.name, "dummy", temp_file2.name],
                    allow_missing_files=True,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=60),
                )
                self.assertEqual(
                    res[0],
                    [
                        (datetime(2022, 1, 1, 0, 0, 1), 1),
                        (datetime(2022, 1, 1, 0, 0, 2), 2),
                        (datetime(2022, 1, 1, 0, 0, 3), 3),
                        (datetime(2022, 1, 1, 0, 0, 4), 4),
                    ],
                )

    def test_bad_field_map(self):
        class MyStruct(csp.Struct):
            x: float

        @csp.graph
        def g():
            reader = ParquetReader([], time_column="timestamp")
            with self.assertRaisesRegex(ValueError, "Invalid field_map type.*"):
                reader.subscribe_all(float, {"x": "x"})

        csp.run(g, starttime=datetime(2022, 1, 1), endtime=timedelta(seconds=10))

    def test_numpy_array_on_struct_with_field_map(self):
        class MyStruct(csp.Struct):
            v1: int
            v2: str
            v3: csp.typing.Numpy1DArray[str]
            v4: csp.typing.Numpy1DArray[int]

            def __eq__(self, other):
                for k in self.metadata():
                    has_attr = hasattr(self, k)
                    if has_attr != hasattr(other, k):
                        return False
                    if has_attr:
                        if k in ("v3", "v4"):
                            if not (getattr(self, k) == getattr(other, k)).all():
                                return False
                        else:
                            if not (getattr(self, k) == getattr(other, k)):
                                return False
                return True

        @csp.graph
        def write_data(file_name: str) -> csp.ts[MyStruct]:
            parquet_writer = ParquetWriter(file_name=file_name, timestamp_column_name="TIME")

            values = csp.curve(
                MyStruct,
                [
                    (timedelta(0), MyStruct(v1=1, v2="my_s", v3=numpy.array(["a", "b", "c"]))),
                    (timedelta(0), MyStruct(v4=numpy.array([1, 2, 3]))),
                    (timedelta(seconds=5), MyStruct(v1=2, v2="my_s")),
                ],
            )
            parquet_writer.publish_struct(
                values, field_map={"v1": "int", "v2": "str", "v3": "np_arr_str", "v4": "np_arr_int"}
            )
            return values

        @csp.graph
        def read_data(file_name: str) -> csp.ts[MyStruct]:
            parquet_reader = ParquetReader(file_name, time_column="TIME")
            field_map = {"np_arr_int": "v4", "np_arr_str": "v3", "str": "v2", "int": "v1"}
            parquet_reader.subscribe_all(MyStruct, field_map=field_map)
            return parquet_reader.subscribe_all(MyStruct, field_map=field_map)

        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            v1 = csp.run(write_data, temp_file.name, starttime=datetime(2022, 1, 1), endtime=timedelta(seconds=10))
            v2 = csp.run(read_data, temp_file.name, starttime=datetime(2022, 1, 1), endtime=timedelta(seconds=10))
            self.assertEqual(v1, v2)

    def test_string_array_with_nulls(self):
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            df = pandas.DataFrame({"v": [["a", None, "b"]], "t": [datetime(2022, 1, 1)]})
            df.to_parquet(temp_file.name)

            def reader_g():
                reader = ParquetReader(temp_file.name, time_column="t", tz=pytz.utc)
                return reader.subscribe_all(csp.typing.Numpy1DArray[str], "v")

            with self.assertRaisesRegex(
                ValueError, ".*Can't read empty value to array from arrow array of type utf8.*"
            ):
                csp.run(reader_g, starttime=datetime(2022, 1, 1), endtime=timedelta(seconds=10))

    def test_float_array_with_nulls(self):
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            df = pandas.DataFrame({"v": [[1.0, None, 2.0]], "t": [datetime(2022, 1, 1)]})
            df.to_parquet(temp_file.name)

            def reader_g():
                reader = ParquetReader(temp_file.name, time_column="t", tz=pytz.utc)
                return reader.subscribe_all(csp.typing.Numpy1DArray[float], "v")

            res = csp.run(reader_g, starttime=datetime(2022, 1, 1), endtime=timedelta(seconds=10))
            res_a = res[0][0][1]
            self.assertEqual(res_a[0], 1)
            self.assertEqual(res_a[2], 2)
            self.assertTrue(math.isnan(res_a[1]))

    def test_all_types(self):
        from datetime import date, time

        class MyEnum(csp.Enum):
            A = 1
            B = 2

        class BaseNative(csp.Struct):
            i: int
            b: bool
            f: float

        class AllTypes(csp.Struct):
            b: bool
            i: int
            d: float
            dt: datetime
            dte: date
            t: time
            s: str
            e: MyEnum

        # timedelta reading isnt supported in parquet, its an open issue with parquet
        # Once sub-struct is fixed ( see link below ) we can remove this intermediate class from the test
        class AllTypesParquet(AllTypes):
            struct: BaseNative

        ## FIXME looks like sub-struct crashes in arrow mode "sub-struct parquet reading crashes on arrow binary"
        # Were avoiding it for now by not including sub struct on the arrow test
        class AllTypesArrow(AllTypes):
            td: timedelta

        @csp.graph
        def write_data(file_name: str, binary_arrow: bool) -> csp.ts[AllTypes]:
            parquet_writer = ParquetWriter(
                file_name=file_name,
                timestamp_column_name="TIME",
                config=ParquetOutputConfig(write_arrow_binary=binary_arrow),
            )

            sType = AllTypesArrow if binary_arrow else AllTypesParquet
            v1 = sType(
                b=True,
                i=123,
                d=123.456,
                dt=datetime.utcnow(),
                dte=date.today(),
                t=time(1, 2, 3),
                s="hello hello",
                e=MyEnum.A,
            )

            if binary_arrow:
                v1.td = timedelta(seconds=0.123)
            else:
                v1.struct = BaseNative(i=456, b=False, f=123.456)

            v2 = v1.copy()
            v2.update(s="xyz", d=456.789, e=MyEnum.B)

            v3 = v2.copy()
            v3.update(i=456)
            if not binary_arrow:
                v3.struct = BaseNative(i=789)

            values = csp.curve(sType, [(timedelta(0), v1), (timedelta(0), v2), (timedelta(seconds=5), v3)])

            for k in sType.metadata().keys():
                if k == "struct" or (not binary_arrow and k == "td"):
                    continue
                elem = getattr(values, k)
                parquet_writer.publish("elem_" + k, elem)

            parquet_writer.publish_struct(values)
            return values

        @csp.node
        def _check_equal(field: str, x: csp.ts[object], y: csp.ts[object]):
            with csp.state():
                s_count = 0

            with csp.stop():
                self.assertGreater(s_count, 0, field)

            if csp.ticked(x, y):
                self.assertEqual(x, y, field)
                s_count += 1

        @csp.graph
        def read_data(file_name: str, binary_arrow: bool) -> csp.ts[AllTypes]:
            parquet_reader = ParquetReader(file_name, time_column="TIME", binary_arrow=binary_arrow)
            sType = AllTypesArrow if binary_arrow else AllTypesParquet

            all_data = parquet_reader.subscribe_all(sType)
            for k, t in sType.metadata().items():
                if k == "struct" or (not binary_arrow and k == "td"):
                    continue
                elem = parquet_reader.subscribe_all(t, k)
                _check_equal(k, getattr(all_data, k), elem)
            return all_data

        for binary_arrow in [False, True]:
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
                temp_file.close()
                v1 = csp.run(
                    write_data,
                    temp_file.name,
                    binary_arrow,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=10),
                )
                v2 = csp.run(
                    read_data,
                    temp_file.name,
                    binary_arrow,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=10),
                )
                self.assertEqual(v1, v2)

    def test_write_metadata(self):
        class MyStruct(csp.Struct):
            a: int
            b: float

        @csp.graph
        def write_data(
            file_name: str, file_metadata: dict, column_metadata: dict, binary_arrow: bool
        ) -> csp.ts[MyStruct]:
            parquet_writer = ParquetWriter(
                file_name=file_name,
                timestamp_column_name="TIME",
                file_metadata=file_metadata,
                column_metadata=column_metadata,
                config=ParquetOutputConfig(write_arrow_binary=binary_arrow),
            )

            values = csp.const(MyStruct(a=1, b=2.0))
            parquet_writer.publish_struct(values)
            return values

        for binary_arrow in [False, True]:
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
                temp_file.close()
                file_metadata = {"some": "text", "foo": "bar"}
                column_metadata = {"a": {"type": "int"}, "b": {"type": "float"}}
                csp.run(
                    write_data,
                    temp_file.name,
                    file_metadata,
                    column_metadata,
                    binary_arrow,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=10),
                )

                if binary_arrow:
                    read_schema = pyarrow.ipc.read_schema(temp_file.name)
                else:
                    read_schema = pyarrow.parquet.read_table(temp_file.name).schema

                # Not quite sure why reading metadata back reads it in as bytes but so be it for the test comparison
                bytes_file_metadata = {bytes(k, "utf-8"): bytes(v, "utf-8") for k, v in file_metadata.items()}
                self.assertEqual(read_schema.metadata, bytes_file_metadata)

                for c, meta in column_metadata.items():
                    bytes_col_metadata = {bytes(k, "utf-8"): bytes(v, "utf-8") for k, v in meta.items()}
                    self.assertEqual(read_schema.field(c).metadata, bytes_col_metadata)

        # Assert exceptions
        with self.assertRaisesRegex(TypeError, "parquet metadata can only have string values"):
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
                temp_file.close()
                csp.run(
                    write_data,
                    temp_file.name,
                    {"a": 1},
                    None,
                    False,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=10),
                )

        with self.assertRaisesRegex(
            TypeError,
            "parquet column metadata can only have string values, got non-string value for metadata on column 'foo'",
        ):
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
                temp_file.close()
                csp.run(
                    write_data,
                    temp_file.name,
                    None,
                    {"foo": {"bar": 1}},
                    False,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=10),
                )

        with self.assertRaisesRegex(
            TypeError,
            "parquet column metadata expects dictionary entry per column, got unrecognized type for column 'foo'",
        ):
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
                temp_file.close()
                csp.run(
                    write_data,
                    temp_file.name,
                    None,
                    {"foo": "bar"},
                    False,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=10),
                )

        with self.assertRaisesRegex(ValueError, "parquet column metadata has unmapped column: 'foo'"):
            with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
                temp_file.close()
                csp.run(
                    write_data,
                    temp_file.name,
                    None,
                    {"foo": {"bar": "baz"}},
                    False,
                    starttime=datetime(2022, 1, 1),
                    endtime=timedelta(seconds=10),
                )

    def test_parquet_int_symbol(self):
        @csp.node
        def _mod(x: csp.ts[int], mod: int) -> csp.ts[int]:
            return x % mod

        @csp.graph
        def g_write(temp_file: object):
            writer = ParquetWriter(temp_file, "csp_timestamp")

            data = csp.count(csp.timer(timedelta(seconds=0.1)))
            symbol = _mod(data, 3)
            writer.publish("symbol", symbol)
            writer.publish("value", data)

        @csp.graph
        def g_read(temp_file: object):
            reader = ParquetReader(temp_file, symbol_column="symbol", time_column="csp_timestamp")
            for symbol in range(3):
                csp.add_graph_output(symbol, reader.subscribe(symbol, int, "value"))

        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            s = datetime(2021, 1, 1)
            e = timedelta(seconds=10)
            csp.run(g_write, temp_file.name, starttime=s, endtime=e)
            res = csp.run(g_read, temp_file.name, starttime=s, endtime=e)
            for sym in range(3):
                self.assertTrue(all(v[1] % 3 == sym for idx, v in enumerate(res[sym])))

    def test_parquet_polars_read(self):
        @csp.graph
        def test_tz(file_name: str, start: datetime):
            data = csp.curve(float, [(start, 1.0)])

            writer = ParquetWriter(
                file_name=file_name, timestamp_column_name="csp_time", config=ParquetOutputConfig(allow_overwrite=True)
            )
            writer.publish("int_vals", data)

        start = datetime(2020, 1, 1, tzinfo=pytz.utc)

        with tempfile.NamedTemporaryFile(suffix=".parquet") as output:
            output.file.close()
            g = csp.run(test_tz, output.name, start, starttime=start)

            struct_df = polars.read_parquet(output.name)

            self.assertEqual(len(struct_df), 1)
            self.assertEqual(struct_df["csp_time"][0].tzinfo.key, "UTC")
            self.assertEqual(struct_df["csp_time"][0].timestamp(), start.timestamp())

    def test_parquet_read(self):
        from csp.impl.types.typing_utils import CspTypingUtils

        def _run_test(dts, items, item_type):
            df = polars.DataFrame({"timestamp": dts, "data": items})
            with tempfile.TemporaryDirectory(prefix="csp_unit_tests") as tmp_folder:
                file_name = os.path.join(tmp_folder, "data.parquet")
                df.write_parquet(file_name)
                table = pyarrow.parquet.read_table(file_name)
                new_schema = pyarrow.schema(
                    [pyarrow.field("timestamp", pyarrow.timestamp("us")), pyarrow.field("data", item_type)]
                )
                pyarrow.parquet.write_table(table.cast(new_schema), file_name)

                @csp.graph
                def my_graph() -> csp.ts[MyStruct]:
                    reader = ParquetReader(file_name, time_column="timestamp")
                    return reader.subscribe_all(MyStruct, MyStruct.default_field_map())

                read_data = csp.run(my_graph, starttime=dts[0], endtime=dts[-1])
            self.assertEqual([tup[0] for tup in read_data[0]], dts, "timestamps don't match")
            structs = [tup[1] for tup in read_data[0]]
            if CspTypingUtils.is_numpy_array_type(MyStruct.__full_metadata_typed__["data"]):
                self.assertEqual([struct.data.tolist() for struct in structs], items)
            else:
                self.assertEqual([struct.data for struct in structs], items)

        NUM_ITEMS = 10
        dts = [datetime.now() + timedelta(seconds=i) for i in range(NUM_ITEMS)]

        # Strings
        class MyStruct(csp.Struct):
            data: str

        _run_test(dts, ["test" for i in range(NUM_ITEMS)], pyarrow.string())
        # Binary
        _run_test(dts, ["test" for i in range(NUM_ITEMS)], pyarrow.binary())
        # Large String
        _run_test(dts, ["test" for i in range(NUM_ITEMS)], pyarrow.large_string())
        # Large Binary
        _run_test(dts, ["test" for i in range(NUM_ITEMS)], pyarrow.large_binary())

        class MyStruct(csp.Struct):
            data: csp.typing.Numpy1DArray[str]

        # List of Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.list_(pyarrow.string()))
        # List of Binary Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.list_(pyarrow.binary()))
        # List of Large Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.list_(pyarrow.large_string()))
        # List of Large Binary Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.list_(pyarrow.large_binary()))
        # Large List of Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.large_list(pyarrow.string()))
        # Large List of Binary Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.large_list(pyarrow.binary()))
        # Large List of Large Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.large_list(pyarrow.large_string()))
        # Large List of Large Binary Strings
        _run_test(dts, [["test"] * i for i in range(NUM_ITEMS)], pyarrow.large_list(pyarrow.large_binary()))

        ## Ints
        class MyStruct(csp.Struct):
            data: int

        # Ints
        _run_test(dts, [i for i in range(NUM_ITEMS)], pyarrow.int64())

        class MyStruct(csp.Struct):
            data: csp.typing.Numpy1DArray[int]

        # List of Ints
        _run_test(dts, [[i] * i for i in range(NUM_ITEMS)], pyarrow.list_(pyarrow.int64()))
        # Large List of Ints
        _run_test(dts, [[i] * i for i in range(NUM_ITEMS)], pyarrow.large_list(pyarrow.int64()))

        ## Floats
        class MyStruct(csp.Struct):
            data: float

        # Floats
        _run_test(dts, [float(i) for i in range(NUM_ITEMS)], pyarrow.float64())

        class MyStruct(csp.Struct):
            data: csp.typing.Numpy1DArray[float]

        # List of Floats
        _run_test(dts, [[float(i)] * i for i in range(NUM_ITEMS)], pyarrow.list_(pyarrow.float64()))
        # Large List of Floats
        _run_test(dts, [[float(i)] * i for i in range(NUM_ITEMS)], pyarrow.large_list(pyarrow.float64()))

        ## Bools
        class MyStruct(csp.Struct):
            data: bool

        # Bool
        _run_test(dts, [True for i in range(NUM_ITEMS)], pyarrow.bool_())

        class MyStruct(csp.Struct):
            data: csp.typing.Numpy1DArray[bool]

        # List of Bools
        _run_test(dts, [[True] * i for i in range(NUM_ITEMS)], pyarrow.list_(pyarrow.bool_()))
        # Large List of Bools
        _run_test(dts, [[True] * i for i in range(NUM_ITEMS)], pyarrow.large_list(pyarrow.bool_()))


if __name__ == "__main__":
    unittest.main()
