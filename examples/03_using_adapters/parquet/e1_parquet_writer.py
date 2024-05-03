import logging
import tempfile
from datetime import datetime, timedelta

import csp
from csp.adapters.parquet import ParquetOutputConfig, ParquetWriter


class Dummy(csp.Struct):
    int_val: int
    float_val: float


@csp.graph
def write_struct(file_name: str):
    st = datetime(2020, 1, 1)

    curve = csp.curve(
        Dummy,
        [
            (st + timedelta(seconds=1), Dummy(int_val=1, float_val=1.0)),
            (st + timedelta(seconds=2), Dummy(int_val=2, float_val=2.0)),
            (st + timedelta(seconds=3), Dummy(int_val=3, float_val=3.0)),
        ],
    )
    writer = ParquetWriter(
        file_name=file_name, timestamp_column_name="csp_time", config=ParquetOutputConfig(allow_overwrite=True)
    )
    writer.publish_struct(curve)


@csp.graph
def write_series(file_name: str):
    st = datetime(2020, 1, 1)

    curve_int = csp.curve(int, [(st + timedelta(seconds=i), i * 5) for i in range(10)])
    curve_str = csp.curve(str, [(st + timedelta(seconds=i), f"str_{i}") for i in range(10)])
    writer = ParquetWriter(
        file_name=file_name, timestamp_column_name="csp_time", config=ParquetOutputConfig(allow_overwrite=True)
    )
    writer.publish("int_vals", curve_int)
    writer.publish("str_vals", curve_str)


@csp.graph
def my_graph(struct_file_name: str, series_file_name: str):
    write_struct(struct_file_name)
    write_series(series_file_name)


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

            try:
                import pandas
            except ModuleNotFoundError:
                logging.warning(
                    "Pandas is not installed, not loading the dataframes, for fully functional example, consider installing pandas"
                )
            else:
                struct_df = pandas.read_parquet(struct_file.name)
                print(f"Struct data:\n{struct_df}")
                series_df = pandas.read_parquet(series_file.name)
                print(f"Series data:\n{series_df}")


if __name__ == "__main__":
    main()
