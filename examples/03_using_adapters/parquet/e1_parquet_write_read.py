import logging
import tempfile
from datetime import datetime, timedelta

import csp
from csp.adapters.parquet import ParquetOutputConfig, ParquetReader, ParquetWriter


class Example(csp.Struct):
    int_val: int
    float_val: float


@csp.graph
def write_struct(file_name: str):
    st = datetime(2020, 1, 1)

    curve = csp.curve(
        Example,
        [
            (st + timedelta(seconds=1), Example(int_val=1, float_val=1.0)),
            (st + timedelta(seconds=2), Example(int_val=2, float_val=2.0)),
            (st + timedelta(seconds=3), Example(int_val=3, float_val=3.0)),
        ],
    )
    writer = ParquetWriter(
        file_name=file_name, timestamp_column_name="csp_time", config=ParquetOutputConfig(allow_overwrite=True)
    )
    writer.publish_struct(curve)


@csp.graph
def write_series(file_name: str):
    st = datetime(2020, 1, 1)

    curve_int = csp.curve(int, [(st + timedelta(seconds=i), i * 5) for i in range(3)])
    curve_float = csp.curve(float, [(st + timedelta(seconds=i), i * 0.1) for i in range(3)])
    writer = ParquetWriter(
        file_name=file_name, timestamp_column_name="csp_time", config=ParquetOutputConfig(allow_overwrite=True)
    )
    writer.publish("int_val", curve_int)
    writer.publish("float_val", curve_float)


@csp.graph
def my_graph(struct_file_name: str, series_file_name: str):
    write_struct(struct_file_name)
    write_series(series_file_name)


@csp.graph
def read_graph(struct_file_name: str, series_file_name: str):
    struct_reader = ParquetReader(struct_file_name, time_column="csp_time")
    struct_all = struct_reader.subscribe_all(Example)
    csp.print("struct_all", struct_all)

    series_reader = ParquetReader(series_file_name, time_column="csp_time")
    series_all = series_reader.subscribe_all(Example)
    csp.print("series_all", series_all)


def main():
    with tempfile.NamedTemporaryFile(suffix=".parquet") as struct_file:
        struct_file.file.close()
        with tempfile.NamedTemporaryFile(suffix=".parquet") as series_file:
            series_file.file.close()
            csp.run(
                my_graph,
                struct_file.name,
                series_file.name,
                starttime=datetime(2020, 1, 1),
                endtime=timedelta(minutes=1),
            )

            print("\nCSP ParquetReader:\n")

            csp.run(
                read_graph,
                struct_file.name,
                series_file.name,
                starttime=datetime(2020, 1, 1),
                endtime=timedelta(minutes=1),
            )

            try:
                import pandas
            except ModuleNotFoundError:
                logging.warning(
                    "Pandas is not installed, not loading the dataframes, for fully functional example, consider installing pandas"
                )
            else:
                struct_df = pandas.read_parquet(struct_file.name)
                print(f"\nStruct pandas DataFrame:\n\n{struct_df}")
                series_df = pandas.read_parquet(series_file.name)
                print(f"\nSeries pandas DataFrame:\n\n{series_df}")


if __name__ == "__main__":
    main()
