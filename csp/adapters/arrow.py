from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

import pyarrow as pa
import pyarrow.parquet as pq
from packaging.version import parse

import csp
from csp.impl.types.tstype import ts
from csp.impl.types.typing_utils import CspTypingUtils
from csp.impl.wiring import input_adapter_def
from csp.lib import _arrowadapterimpl

__all__ = [
    "CRecordBatchPullInputAdapter",
    "RecordBatchPullInputAdapter",
    "record_batches_to_struct",
    "struct_to_record_batches",
    "write_record_batches",
]

_PYARROW_HAS_CONCAT_BATCHES = parse(pa.__version__) >= parse("19.0.0")

T = TypeVar("T")


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
    ts_col_name: str,
    source: Iterable[pa.RecordBatch],
    schema: Optional[pa.Schema] = None,
    expect_small_batches: bool = False,
) -> csp.ts[[pa.RecordBatch]]:
    """Stream record batches from an iterator/generator into csp

    Args:
        ts_col_name: Name of the timestamp column containing timestamps in ascending order
        source: Iterator/generator of record batches
        schema: Schema of the record batches. If None, extracted from first batch at runtime.
        expect_small_batches: Optional flag to optimize performance for scenarios where there are few rows (<10) per timestamp

    NOTE: The ascending order of the timestamp column must be enforced by the caller
    """
    # Validate only if schema provided upfront
    if schema is not None:
        ts_col = schema.field(ts_col_name)
        if not pa.types.is_timestamp(ts_col.type):
            raise ValueError(f"{ts_col_name} is not a valid timestamp column in the schema")
        c_schema = schema.__arrow_c_schema__()
    else:
        c_schema = None  # C++ will extract from first batch

    c_source = map(lambda rb: rb.__arrow_c_array__(), source)
    c_data = CRecordBatchPullInputAdapter(ts_col_name, c_source, c_schema, expect_small_batches)
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


def _extract_numpy_metadata(cls, field_map, numpy_dimensions_column_map):
    """Categorize struct fields into scalar and numpy fields.

    Args:
        cls: csp.Struct type
        field_map: Resolved mapping of struct field name -> arrow column name (must not be None).
        numpy_dimensions_column_map: Mapping of arrow column name -> dimensions column name.

    Returns:
        Tuple of (scalar_fields, numpy_fields, numpy_dimension_names, numpy_element_types) where:
        - scalar_fields: list of (struct_field_name, arrow_col_name) tuples
        - numpy_fields: dict {arrow_col_name: struct_field_name}
        - numpy_dimension_names: dict {arrow_col_name: dim_col_name}
        - numpy_element_types: dict {struct_field_name: python_type}
    """
    from csp.adapters.output_adapters.parquet import resolve_array_shape_column_name

    numpy_dimensions_column_map = numpy_dimensions_column_map or {}
    meta_typed = cls.metadata(typed=True)
    scalar_fields = []
    numpy_fields = {}
    numpy_element_types = {}
    numpy_dimension_names = {}

    for struct_field_name, arrow_col_name in field_map.items():
        field_typ = meta_typed[struct_field_name]
        if CspTypingUtils.is_numpy_array_type(field_typ):
            numpy_fields[arrow_col_name] = struct_field_name
            numpy_element_types[struct_field_name] = field_typ.__args__[0]

            if CspTypingUtils.is_numpy_nd_array_type(field_typ):
                dim_col_name = resolve_array_shape_column_name(
                    arrow_col_name, numpy_dimensions_column_map.get(arrow_col_name, None)
                )
                numpy_dimension_names[arrow_col_name] = dim_col_name
        else:
            scalar_fields.append((struct_field_name, arrow_col_name))

    return scalar_fields, numpy_fields, numpy_dimension_names, numpy_element_types


@csp.node(cppimpl=_arrowadapterimpl.record_batches_to_struct)
def _record_batches_to_struct(
    schema_ptr: object,
    cls: "T",
    properties: dict,
    data: ts[object],
) -> ts[List["T"]]:
    raise NotImplementedError("C++ implementation only")
    return None


@csp.graph
def record_batches_to_struct(
    data: ts[List[pa.RecordBatch]],
    cls: "T",
    schema: pa.Schema,
    field_map: Optional[Dict[str, str]] = None,
    numpy_dimensions_column_map: Optional[Dict[str, str]] = None,
) -> ts[List["T"]]:
    """Convert ts[List[pa.RecordBatch]] into ts[List[T]] where T is a csp.Struct type.

    Args:
        data: Timeseries of lists of Arrow RecordBatches
        cls: Target csp.Struct type
        schema: Arrow schema of the record batches (required).
        field_map: Optional mapping of struct field name -> arrow column name.
            If None, defaults to identity naming for all fields.
        numpy_dimensions_column_map: Optional mapping of arrow column name -> dimensions column name
            for NumpyNDArray fields. If not provided for an NDArray field, defaults to
            ``<column_name>_csp_dimensions``.

    Returns:
        Timeseries of lists of struct instances
    """
    if field_map is None:
        field_map = {name: name for name in cls.metadata(typed=True)}

    scalar_fields, numpy_fields, numpy_dimension_names, _ = _extract_numpy_metadata(
        cls, field_map, numpy_dimensions_column_map
    )
    scalar_field_map = {arrow_col: struct_field for struct_field, arrow_col in scalar_fields}

    # Build properties dict for the C++ node
    properties = {
        "field_map": scalar_field_map,
        "numpy_fields": numpy_fields,
        "numpy_dimension_names": numpy_dimension_names,
    }

    # Export schema to PyCapsule
    schema_capsule = schema.__arrow_c_schema__()

    # Convert RecordBatches to PyCapsule tuples for C++ consumption
    c_data = csp.apply(
        data,
        lambda batches: [rb.__arrow_c_array__() for rb in batches],
        object,
    )

    return _record_batches_to_struct(schema_capsule, cls, properties, c_data)


@csp.node(cppimpl=_arrowadapterimpl.struct_to_record_batches)
def _struct_to_record_batches(
    cls: "T",
    properties: dict,
    data: ts[List["T"]],
) -> ts[object]:
    raise NotImplementedError("C++ implementation only")
    return None


@csp.graph
def struct_to_record_batches(
    data: ts[List["T"]],
    cls: "T",
    field_map: Optional[Dict[str, str]] = None,
    numpy_dimensions_column_map: Optional[Dict[str, str]] = None,
    max_batch_size: int = 65536,
) -> ts[List[pa.RecordBatch]]:
    """Convert ts[List[T]] into ts[List[pa.RecordBatch]] where T is a csp.Struct type.

    Args:
        data: Timeseries of lists of struct instances
        cls: Source csp.Struct type
        field_map: Optional mapping of struct field name -> arrow column name.
            If None, defaults to identity naming for all fields.
        numpy_dimensions_column_map: Optional mapping of arrow column name -> dimensions column name
            for NumpyNDArray fields. If not provided for an NDArray field, defaults to
            ``<column_name>_csp_dimensions``.
        max_batch_size: Maximum number of rows per output RecordBatch.
            Defaults to 65536. Set to 0 to disable chunking.

    Returns:
        Timeseries of lists of RecordBatch
    """
    if field_map is None:
        field_map = {name: name for name in cls.metadata(typed=True)}

    scalar_fields, numpy_fields, numpy_dimension_names, numpy_element_types = _extract_numpy_metadata(
        cls, field_map, numpy_dimensions_column_map
    )
    scalar_field_map = {struct_field: arrow_col for struct_field, arrow_col in scalar_fields}

    # Build properties dict for the C++ node
    properties = {
        "field_map": scalar_field_map,
        "numpy_fields": numpy_fields,
        "numpy_element_types": numpy_element_types,
        "numpy_dimension_names": numpy_dimension_names,
        "max_batch_size": max_batch_size,
    }

    # Call C++ node, then convert capsule tuples -> RecordBatch
    c_data = _struct_to_record_batches(cls, properties, data)

    return csp.apply(
        c_data,
        lambda c_tups: [pa.record_batch(_RecordBatchCSource(c_tup)) for c_tup in c_tups],
        List[pa.RecordBatch],
    )
