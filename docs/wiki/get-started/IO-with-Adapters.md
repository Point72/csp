In [First Steps](First-Steps) and [More with CSP](More-with-CSP) we used toy data for the streaming workflows. In real workflows, we need to access data stored in particular storage formats. To bring data into or out of a `csp` graph, we use **adapters**.

`csp` has [several built-in adapters to access certain types of data](Input-Output-Adapters-API) such as Kafka and Parquet. You can also write your own adapters for any other data format; for reference, see the various "How-to" guides for [historical](Write-Historical-Input-Adapters), [real-time](Write-Realtime-Input-Adapters) and [output](Write-Output-Adapters) adapters. I/O adapters form the interface between external data and the time series format used in `csp`.

In this tutorial, you write to and read from Parquet files on the local file system.

`csp` has the `ParquetWriter` and `ParquetReader` adapters to stream data to and from Parquet files. Check out the complete [API in the Reference documentation](https://github.com/Point72/csp/wiki/Input-Output-Adapters-API#parquet).

> \[!IMPORTANT\]
> `csp` graphs can process historical *and* real-time data with little to no changes in the application code.

## Streaming a csp.Struct

A `csp.Struct` is a basic form of structured data in `csp` where each field can be accessed as its own time series. It is analogous to a dataclass in Python, and its fields must be type annotated. We will store some example data in a custom struct called `Example` and then stream the struct to a Parquet file.

```python
from csp.adapters.parquet import ParquetOutputConfig, ParquetWriter, ParquetReader

class Example(csp.Struct):
    int_val: int
    float_val: float
```

## Write to a Parquet file

In a graph, create some sample values for `Example` and use `ParquetWriter` to stream to a Parquet file.

1. The `timestamp_column_name` is how `csp` preserves the timestamps on each event. If the timestamp information is not required, you can set the column name argument to `None`.
1. You can provide optional configurations in the `ParquetOutputConfig` format which can set `allow_overwrite`, `batch_size`, `compression`, and `write_arrow_binary`.
1. We use `publish_struct` to publish (write) the time series data to disk.

```python
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
```

## Read from Parquet file

You can use `ParquetReader` with a `time_column` to read back the data.

```python
@csp.graph
def read_struct(file_name: str):
    struct_reader = ParquetReader(file_name, time_column="csp_time")
    struct_all = struct_reader.subscribe_all(Example)
    csp.print("struct_all", struct_all)
```

Go through the complete example at [examples/03_using_adapters/parquet/e1_parquet_write_read.py](https://github.com/Point72/csp/blob/main/examples/03_using_adapters/parquet/e1_parquet_write_read.py) and check out the the [API reference](Input-Output-Adapters-API#parquet) for more details.
