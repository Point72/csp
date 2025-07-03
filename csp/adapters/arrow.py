from typing import Iterable, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
from packaging.version import parse

import csp
from csp.impl.types.tstype import ts
from csp.impl.wiring import input_adapter_def
from csp.lib import _arrowadapterimpl

__all__ = [
    "CRecordBatchPullInputAdapter",
    "RecordBatchPullInputAdapter",
    "write_record_batches",
]

_PYARROW_HAS_CONCAT_BATCHES = parse(pa.__version__) >= parse("19.0.0")


CRecordBatchPullInputAdapter = input_adapter_def(
    "CRecordBatchPullInputAdapter",
    _arrowadapterimpl._record_batch_input_adapter_creator,
    ts[List[Tuple[object, object]]],
    ts_col_name=str,
    source=Iterable[Tuple[object, object]],
    schema=object,
    expect_small_batches=bool,
)
"""Stream record batches using the PyCapsule C Data interface from an iterator/generator into csp

Args:
    ts_col_name: Name of the timestamp column containing timestamps in ascending order
    source: Iterator/generator of pycapsule objects obtained by calling __arrow_c_array__() on the python record batches
    schema: The schema of the record batches
    expect_small_batches: Optional flag to optimize performance for scenarios where there are few rows (<10) per timestamp

Returns:
    List of pycapsule objects each corresponding to a record batch in the PyCapsule representation (similar to calling __arrow_c_array__() on a record batch)

NOTE: The ascending order of the timestamp column must be enforced by the caller
"""


class _RecordBatchCSource:
    def __init__(self, tup):
        self.tup = tup

    def __arrow_c_array__(self, requested_schema=None):
        return self.tup


@csp.graph
def RecordBatchPullInputAdapter(
    ts_col_name: str, source: Iterable[pa.RecordBatch], schema: pa.Schema, expect_small_batches: bool = False
) -> csp.ts[[pa.RecordBatch]]:
    """Stream record batches from an iterator/generator into csp

    Args:
        ts_col_name: Name of the timestamp column containing timestamps in ascending order
        source: Iterator/generator of record batches
        schema: The schema of the record batches
        expect_small_batches: Optional flag to optimize performance for scenarios where there are few rows (<10) per timestamp

    NOTE: The ascending order of the timestamp column must be enforced by the caller
    """
    # Safety checks
    ts_col = schema.field(ts_col_name)
    if not pa.types.is_timestamp(ts_col.type):
        raise ValueError(f"{ts_col_name} is not a valid timestamp column in the schema")

    c_source = map(lambda rb: rb.__arrow_c_array__(), source)
    c_data = CRecordBatchPullInputAdapter(ts_col_name, c_source, schema.__arrow_c_schema__(), expect_small_batches)
    return csp.apply(
        c_data, lambda c_tups: [pa.record_batch(_RecordBatchCSource(c_tup)) for c_tup in c_tups], List[pa.RecordBatch]
    )


def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    if _PYARROW_HAS_CONCAT_BATCHES:
        # pyarrow version 19+ support concat_batches API
        return pa.concat_batches(batches)
    else:
        combined_table = pa.Table.from_batches(batches).combine_chunks()
        combined_batches = combined_table.to_batches()
        if len(combined_batches) > 1:
            raise ValueError("Not able to combine multiple record batches into one record batch")
        return combined_batches[0]


@csp.node
def write_record_batches(
    where: str,
    batches: csp.ts[List[pa.RecordBatch]],
    kwargs: dict,
    merge_record_batches: bool = False,
    max_batch_size: int = 0,
):
    """
    Dump all the record batches to a parquet file

    Args:
        where: destination to write the data to
        batches: The timeseries of list of record batches
        kwargs: additional args to pass to the ParquetWriter
        merge_record_batches: A flag to combine all the record batches in a single tick into a single record batch
        max_batch_size: the max size of each batch to be written, combine record batches across ticks, does not split record batches. So if a record batch is larger than max_batch_size, it is written as is.
    """
    with csp.state():
        s_writer = None
        s_destination = where
        s_merge_batches = merge_record_batches
        s_max_batch_size = max_batch_size
        s_prev_batch = None
        s_prev_batch_size = 0

    with csp.stop():
        if s_writer:
            if s_prev_batch:
                s_writer.write_batch(_concat_batches(s_prev_batch))
            s_writer.close()

    if csp.ticked(batches):
        if s_merge_batches:
            batches = [_concat_batches(batches)]

        for batch in batches:
            if len(batch) == 0:
                continue
            if s_writer is None:
                s_writer = pq.ParquetWriter(s_destination, batch.schema, **kwargs)
            if s_prev_batch is None:
                s_prev_batch = [batch]
                s_prev_batch_size = len(batch)
            elif s_prev_batch_size + len(batch) > s_max_batch_size:
                s_writer.write_batch(_concat_batches(s_prev_batch))
                s_prev_batch = [batch]
                s_prev_batch_size = len(batch)
            else:
                s_prev_batch += [batch]
                s_prev_batch_size += len(batch)
