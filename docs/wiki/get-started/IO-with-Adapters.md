> \[!WARNING\]
> This page is a work in progress.

In [First Steps](First-Steps) and [More with csp](More-with-CSP) you created example/sample data for the streaming workflows. In actual use-cases, you will use data stored in particular formats and storage spaces, that can be accessed directly or through an API.

csp has [several built-in "adapters" to access certain types of data](https://github.com/Point72/csp/wiki/Input-Output-Adapters-API), including Kafka and Parquet. In this tutorial, you read and write Parquet files on your local file system.

> \[!NOTE\]
> csp can handle historical and real-time data, and the csp program remains similar in both cases.

```python
from csp.adapters.parquet import ParquetOutputConfig, ParquetWriter, ParquetReader

class Dummy(csp.Struct):
    int_val: int
    float_val: float
```

Write a parquet file:

```python
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
```

Read the parquet file:

```python
@csp.graph
def read_struct(file_name: str):
    reader = ParquetReader(
        file_name, time_column="csp_time"
    )
    values = reader.subscribe_all(typ=Dummy)
    csp.print(...)
```

<!-- TODO, add note about https://github.com/melissawm/csp/blob/add-wikimedia-example/examples/02_intermediate/wikimedia.ipynb -->
