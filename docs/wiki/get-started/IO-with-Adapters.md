In [First Steps](First-Steps) and [More with csp](More-with-CSP) you created example/sample data for the streaming workflows. In real workflows, you use data stored in particular formats and storage spaces, that can be accessed directly or through an API.

csp has [several built-in "adapters" to access certain types of data](Input-Output-Adapters-API), including Kafka and Parquet. csp requires friendly (Time Series) data types, and the I/O adapters form the interface between the data types. You can also write your own adapters for other data types, check the corresponding how-to guides for more details.

In this tutorial, you write to, and read from, Parquet files on the local file system.

csp has the `ParquetWriter` and `ParquetReader` adapters to stream data to and from Parquet files. Check out the complete [API in the Reference documentation](https://github.com/Point72/csp/wiki/Input-Output-Adapters-API#parquet).

> \[!IMPORTANT\]
> csp can handle historical and real-time data, and the csp program remains similar in both cases.

## Example

You start with an Example `csp.Struct` data type which will be streamed into a Parquet file.

```python
from csp.adapters.parquet import ParquetOutputConfig, ParquetWriter, ParquetReader

class Example(csp.Struct):
    int_val: int
    float_val: float
```

## Write to a Parquet file

In a graph, create some sample values for `Example` and use `ParquetWriter` to stream it to a Parquet file.

1. The `timestamp_column_name` is how csp preserves the timestamps. Since you need to read this file back into csp, you can provide a column name. If this was the final output and the time stamp information is not required, you can provide `None`.

1. You can provide optional configurations to `config` in the `ParquetOutputConfig` format (which can set `allow_overwrite`, , `batch_size`, `compression`, `write_arrow_binary`).

1. Use `publish_struct` to publish (write) the data as Time Series.

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

You can use `ParquetReader` with a `time_column`, and read all the `Example` rows with `subscribe_all`.

```python
@csp.graph
def read_struct(file_name: str):
    struct_reader = ParquetReader(file_name, time_column="csp_time")
    struct_all = struct_reader.subscribe_all(Example)
    csp.print("struct_all", struct_all)
```

Go through the complete example at [examples/03_using_adapters/parquet/e1_parquet_write_read.py](https://github.com/Point72/csp/blob/main/examples/03_using_adapters/parquet/e1_parquet_write_read.py) and check out the the [API reference](Input-Output-Adapters-API#parquet) for more details.
