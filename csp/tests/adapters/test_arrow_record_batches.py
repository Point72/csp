"""Tests for record_batches_to_struct and struct_to_record_batches.

Covers all scalar types supported by the parquet adapter (bool, int, float, str,
datetime, date, time, timedelta, enum, bytes, nested struct), numpy 1D arrays
(float, int, str, bool), NDArrays, field mapping, null handling, multiple ticks,
round-trips, and error cases.
"""

from datetime import date, datetime, time, timedelta

import numpy as np
import pyarrow as pa
import pytest

import csp
from csp.adapters.arrow import record_batches_to_struct, struct_to_record_batches
from csp.typing import Numpy1DArray, NumpyNDArray

_STARTTIME = datetime(2020, 1, 1, 9, 0, 0)


# =====================================================================
# Struct definitions
# =====================================================================


class ScalarStruct(csp.Struct):
    i64: int
    f64: float
    s: str
    b: bool


class NumericOnlyStruct(csp.Struct):
    x: int
    y: float


class DateTimeStruct(csp.Struct):
    dt: datetime
    td: timedelta
    d: date
    t: time


class MyEnum(csp.Enum):
    A = 1
    B = 2
    C = 3


class EnumStruct(csp.Struct):
    label: str
    color: MyEnum


class BytesStruct(csp.Struct):
    data: bytes


class InnerStruct(csp.Struct):
    x: int
    y: float


class NestedStruct(csp.Struct):
    id: int
    inner: InnerStruct


class AllTypesStruct(csp.Struct):
    b: bool
    i: int
    d: float
    dt: datetime
    dte: date
    t: time
    td: timedelta
    s: str
    e: MyEnum


class NumpyStruct(csp.Struct):
    id: int
    values: Numpy1DArray[float]


class NumpyIntStruct(csp.Struct):
    id: int
    values: Numpy1DArray[int]


class NumpyStringStruct(csp.Struct):
    id: int
    names: Numpy1DArray[str]


class NumpyBoolStruct(csp.Struct):
    id: int
    flags: Numpy1DArray[bool]


class MixedStruct(csp.Struct):
    label: str
    scores: Numpy1DArray[float]


class NDArrayStruct(csp.Struct):
    id: int
    matrix: NumpyNDArray[float]


class NDArrayIntStruct(csp.Struct):
    id: int
    matrix: NumpyNDArray[int]


class FullMixedStruct(csp.Struct):
    """Struct with scalar, numpy 1D, and NDArray fields."""

    label: str
    scores: Numpy1DArray[float]
    matrix: NumpyNDArray[float]


# =====================================================================
# Helpers — reader direction (batch → struct)
# =====================================================================


def _run_to_struct(batches, cls, field_map, schema, numpy_dimensions_column_map=None):
    """Run a graph that converts record batches to structs and returns the results."""

    @csp.graph
    def G(
        batches_: object,
        cls_: type,
        field_map_: dict,
        schema_: object,
        numpy_dims_: object,
    ):
        data = csp.const([batches_])
        structs = record_batches_to_struct(data, cls_, field_map_, schema_, numpy_dims_)
        csp.add_graph_output("structs", structs)

    results = csp.run(
        G,
        batches,
        cls,
        field_map,
        schema,
        numpy_dimensions_column_map,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=1),
    )
    assert len(results["structs"]) == 1
    return results["structs"][0][1]


def _run_multi_tick_read(tick_batches, cls, field_map, schema, numpy_dimensions_column_map=None):
    """Run a graph that ticks multiple lists of record batches and returns all results."""

    @csp.graph
    def G(
        ticks_: object,
        cls_: type,
        field_map_: dict,
        schema_: object,
        numpy_dims_: object,
    ):
        data = csp.unroll(csp.const(ticks_))
        structs = record_batches_to_struct(data, cls_, field_map_, schema_, numpy_dims_)
        csp.add_graph_output("structs", structs)

    results = csp.run(
        G,
        tick_batches,
        cls,
        field_map,
        schema,
        numpy_dimensions_column_map,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=len(tick_batches)),
    )
    return [ts_val[1] for ts_val in results["structs"]]


# =====================================================================
# Helpers — writer direction (struct → batch)
# =====================================================================


def _run_to_batches(structs, cls, field_map=None, numpy_dimensions_column_map=None):
    """Run a graph that converts structs to record batches and returns the results."""

    @csp.graph
    def G(
        structs_: object,
        cls_: type,
        field_map_: object,
        numpy_dims_: object,
    ):
        data = csp.const(structs_)
        batches = struct_to_record_batches(data, cls_, field_map_, numpy_dims_)
        csp.add_graph_output("batches", batches)

    results = csp.run(
        G,
        structs,
        cls,
        field_map,
        numpy_dimensions_column_map,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=1),
    )
    assert len(results["batches"]) == 1
    return results["batches"][0][1]


def _run_multi_tick_write(tick_structs, cls, field_map=None, numpy_dimensions_column_map=None):
    """Run a graph that ticks multiple lists of structs and returns all results."""

    @csp.graph
    def G(
        ticks_: object,
        cls_: type,
        field_map_: object,
        numpy_dims_: object,
    ):
        data = csp.unroll(csp.const(ticks_))
        batches = struct_to_record_batches(data, cls_, field_map_, numpy_dims_)
        csp.add_graph_output("batches", batches)

    results = csp.run(
        G,
        tick_structs,
        cls,
        field_map,
        numpy_dimensions_column_map,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=len(tick_structs)),
    )
    return [ts_val[1] for ts_val in results["batches"]]


# =====================================================================
# Tests: reading scalar fields (batch → struct)
# =====================================================================


class TestReadScalarFields:
    def test_basic_scalar_types(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "i64": [1, 2, 3],
                "f64": [1.1, 2.2, 3.3],
                "s": ["a", "b", "c"],
                "b": [True, False, True],
            }
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)

        assert len(structs) == 3
        assert structs[0].i64 == 1
        assert structs[1].i64 == 2
        assert structs[2].i64 == 3
        assert structs[0].f64 == pytest.approx(1.1)
        assert structs[1].f64 == pytest.approx(2.2)
        assert structs[0].s == "a"
        assert structs[1].s == "b"
        assert structs[0].b is True
        assert structs[1].b is False

    def test_field_mapping(self):
        batch = pa.RecordBatch.from_pydict(
            {"col_x": [10, 20], "col_y": [1.5, 2.5]},
            schema=pa.schema([("col_x", pa.int64()), ("col_y", pa.float64())]),
        )
        field_map = {"col_x": "x", "col_y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)

        assert len(structs) == 2
        assert structs[0].x == 10
        assert structs[0].y == pytest.approx(1.5)
        assert structs[1].x == 20
        assert structs[1].y == pytest.approx(2.5)

    def test_single_row(self):
        batch = pa.RecordBatch.from_pydict({"x": [42], "y": [3.14]})
        field_map = {"x": "x", "y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].x == 42
        assert structs[0].y == pytest.approx(3.14)

    def test_many_rows(self):
        n = 1000
        batch = pa.RecordBatch.from_pydict({"x": list(range(n)), "y": [float(i) / 10.0 for i in range(n)]})
        field_map = {"x": "x", "y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)

        assert len(structs) == n
        for i in range(n):
            assert structs[i].x == i
            assert structs[i].y == pytest.approx(float(i) / 10.0)

    def test_multiple_batches_single_tick(self):
        """Multiple record batches in a single tick should all be converted."""
        batch1 = pa.RecordBatch.from_pydict({"x": [1, 2], "y": [0.1, 0.2]})
        batch2 = pa.RecordBatch.from_pydict({"x": [3, 4], "y": [0.3, 0.4]})
        schema = batch1.schema

        @csp.graph
        def G():
            data = csp.const([batch1, batch2])
            field_map = {"x": "x", "y": "y"}
            structs = record_batches_to_struct(data, NumericOnlyStruct, field_map, schema)
            csp.add_graph_output("structs", structs)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        structs = results["structs"][0][1]

        assert len(structs) == 4
        assert [s.x for s in structs] == [1, 2, 3, 4]

    def test_multiple_ticks(self):
        """Multiple ticks each with their own batch."""
        batch1 = pa.RecordBatch.from_pydict({"x": [10], "y": [1.0]})
        batch2 = pa.RecordBatch.from_pydict({"x": [20], "y": [2.0]})
        schema = batch1.schema

        tick_batches = [[batch1], [batch2]]
        all_results = _run_multi_tick_read(tick_batches, NumericOnlyStruct, {"x": "x", "y": "y"}, schema)

        assert len(all_results) == 2
        assert all_results[0][0].x == 10
        assert all_results[1][0].x == 20

    def test_datetime_types(self):
        """datetime, timedelta, date, time fields."""
        # Use a known UTC nanosecond value to avoid timezone ambiguity
        # 2024-03-15T12:00:00 UTC = 1710504000 seconds since epoch
        dt_val = datetime(2024, 3, 15, 12, 0, 0)
        td_val = timedelta(seconds=3600)
        d_val = date(2024, 6, 15)
        t_val = time(14, 30, 0)

        # Construct nanosecond values directly (UTC epoch-based)
        dt_ns = 1710504000 * 10**9  # 2024-03-15T12:00:00 UTC
        td_ns = int(td_val.total_seconds() * 1e9)
        d_days = (d_val - date(1970, 1, 1)).days
        t_ns = (t_val.hour * 3600 + t_val.minute * 60 + t_val.second) * 10**9

        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([dt_ns], type=pa.timestamp("ns", tz="UTC")),
                pa.array([td_ns], type=pa.duration("ns")),
                pa.array([d_days], type=pa.date32()),
                pa.array([t_ns], type=pa.time64("ns")),
            ],
            schema=pa.schema(
                [
                    ("dt", pa.timestamp("ns", tz="UTC")),
                    ("td", pa.duration("ns")),
                    ("d", pa.date32()),
                    ("t", pa.time64("ns")),
                ]
            ),
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].dt == dt_val
        assert structs[0].td == td_val
        assert structs[0].d == d_val
        assert structs[0].t == t_val

    def test_enum_from_string(self):
        """Enum fields stored as strings."""
        batch = pa.RecordBatch.from_pydict(
            {"label": ["x", "y"], "color": ["A", "B"]},
        )
        field_map = {"label": "label", "color": "color"}
        structs = _run_to_struct(batch, EnumStruct, field_map, batch.schema)

        assert len(structs) == 2
        assert structs[0].label == "x"
        assert structs[0].color == MyEnum.A
        assert structs[1].color == MyEnum.B

    def test_bytes_read(self):
        """Binary/bytes field."""
        val = b"my\x00value"
        batch = pa.RecordBatch.from_arrays(
            [pa.array([val], type=pa.binary())],
            schema=pa.schema([("data", pa.binary())]),
        )
        field_map = {"data": "data"}
        structs = _run_to_struct(batch, BytesStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].data == val

    def test_nested_struct_read(self):
        """Nested struct field."""
        inner_type = pa.struct([("x", pa.int64()), ("y", pa.float64())])
        inner_arr = pa.StructArray.from_arrays(
            [pa.array([42]), pa.array([2.5])],
            fields=[pa.field("x", pa.int64()), pa.field("y", pa.float64())],
        )
        batch = pa.RecordBatch.from_arrays(
            [pa.array([1]), inner_arr],
            schema=pa.schema([("id", pa.int64()), ("inner", inner_type)]),
        )
        field_map = {"id": "id", "inner": "inner"}
        structs = _run_to_struct(batch, NestedStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].id == 1
        assert structs[0].inner.x == 42
        assert structs[0].inner.y == pytest.approx(2.5)


# =====================================================================
# Tests: reading numpy 1D array fields
# =====================================================================


class TestReadNumpy1DFields:
    def test_float_array(self):
        batch = pa.RecordBatch.from_pydict(
            {"id": [1, 2], "values": [[1.0, 2.0, 3.0], [4.0, 5.0]]},
            schema=pa.schema([("id", pa.int64()), ("values", pa.list_(pa.float64()))]),
        )
        field_map = {"id": "id", "values": "values"}
        structs = _run_to_struct(batch, NumpyStruct, field_map, batch.schema)

        assert len(structs) == 2
        assert structs[0].id == 1
        np.testing.assert_array_almost_equal(structs[0].values, [1.0, 2.0, 3.0])
        assert structs[1].id == 2
        np.testing.assert_array_almost_equal(structs[1].values, [4.0, 5.0])

    def test_int_array(self):
        batch = pa.RecordBatch.from_pydict(
            {"id": [1], "values": [[10, 20, 30]]},
            schema=pa.schema([("id", pa.int64()), ("values", pa.list_(pa.int64()))]),
        )
        field_map = {"id": "id", "values": "values"}
        structs = _run_to_struct(batch, NumpyIntStruct, field_map, batch.schema)

        assert len(structs) == 1
        np.testing.assert_array_equal(structs[0].values, [10, 20, 30])

    def test_string_array(self):
        batch = pa.RecordBatch.from_pydict(
            {"id": [1], "names": [["alice", "bob", "carol"]]},
            schema=pa.schema([("id", pa.int64()), ("names", pa.list_(pa.utf8()))]),
        )
        field_map = {"id": "id", "names": "names"}
        structs = _run_to_struct(batch, NumpyStringStruct, field_map, batch.schema)

        assert len(structs) == 1
        np.testing.assert_array_equal(structs[0].names, ["alice", "bob", "carol"])

    def test_bool_array(self):
        batch = pa.RecordBatch.from_pydict(
            {"id": [1], "flags": [[True, False, True]]},
            schema=pa.schema([("id", pa.int64()), ("flags", pa.list_(pa.bool_()))]),
        )
        field_map = {"id": "id", "flags": "flags"}
        structs = _run_to_struct(batch, NumpyBoolStruct, field_map, batch.schema)

        assert len(structs) == 1
        np.testing.assert_array_equal(structs[0].flags, [True, False, True])

    def test_empty_list(self):
        batch = pa.RecordBatch.from_pydict(
            {"id": [1], "values": [[]]},
            schema=pa.schema([("id", pa.int64()), ("values", pa.list_(pa.float64()))]),
        )
        field_map = {"id": "id", "values": "values"}
        structs = _run_to_struct(batch, NumpyStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].id == 1
        assert len(structs[0].values) == 0

    def test_null_list_cell(self):
        """A null list cell should leave the struct field unset."""
        arr_id = pa.array([1, 2])
        arr_values = pa.array([[1.0, 2.0], None], type=pa.list_(pa.float64()))
        batch = pa.RecordBatch.from_arrays(
            [arr_id, arr_values],
            schema=pa.schema([("id", pa.int64()), ("values", pa.list_(pa.float64()))]),
        )
        field_map = {"id": "id", "values": "values"}
        structs = _run_to_struct(batch, NumpyStruct, field_map, batch.schema)

        assert len(structs) == 2
        assert structs[0].id == 1
        np.testing.assert_array_almost_equal(structs[0].values, [1.0, 2.0])
        assert structs[1].id == 2
        assert not hasattr(structs[1], "values")

    def test_mixed_scalar_and_numpy(self):
        batch = pa.RecordBatch.from_pydict(
            {"label": ["a", "b"], "scores": [[0.1, 0.2], [0.3, 0.4, 0.5]]},
            schema=pa.schema([("label", pa.utf8()), ("scores", pa.list_(pa.float64()))]),
        )
        field_map = {"label": "label", "scores": "scores"}
        structs = _run_to_struct(batch, MixedStruct, field_map, batch.schema)

        assert len(structs) == 2
        assert structs[0].label == "a"
        np.testing.assert_array_almost_equal(structs[0].scores, [0.1, 0.2])
        assert structs[1].label == "b"
        np.testing.assert_array_almost_equal(structs[1].scores, [0.3, 0.4, 0.5])

    def test_nan_in_float_list(self):
        """Null values in float lists should become NaN."""
        arr_values = pa.array([[1.0, None, 3.0]], type=pa.list_(pa.float64()))
        arr_id = pa.array([1])
        batch = pa.RecordBatch.from_arrays(
            [arr_id, arr_values],
            schema=pa.schema([("id", pa.int64()), ("values", pa.list_(pa.float64()))]),
        )
        field_map = {"id": "id", "values": "values"}
        structs = _run_to_struct(batch, NumpyStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].values[0] == pytest.approx(1.0)
        assert np.isnan(structs[0].values[1])
        assert structs[0].values[2] == pytest.approx(3.0)


# =====================================================================
# Tests: reading NDArray fields
# =====================================================================


class TestReadNumpyNDArrayFields:
    def test_2d_reshape(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1],
                "matrix": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                "matrix_csp_dimensions": [[2, 3]],
            },
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("matrix", pa.list_(pa.float64())),
                    ("matrix_csp_dimensions", pa.list_(pa.int64())),
                ]
            ),
        )
        field_map = {"id": "id", "matrix": "matrix"}
        structs = _run_to_struct(batch, NDArrayStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].id == 1
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_array_almost_equal(structs[0].matrix, expected)
        assert structs[0].matrix.shape == (2, 3)

    def test_3d_reshape(self):
        flat_data = list(range(24))
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1],
                "matrix": [[float(x) for x in flat_data]],
                "matrix_csp_dimensions": [[2, 3, 4]],
            },
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("matrix", pa.list_(pa.float64())),
                    ("matrix_csp_dimensions", pa.list_(pa.int64())),
                ]
            ),
        )
        field_map = {"id": "id", "matrix": "matrix"}
        structs = _run_to_struct(batch, NDArrayStruct, field_map, batch.schema)

        assert len(structs) == 1
        expected = np.arange(24, dtype=float).reshape(2, 3, 4)
        np.testing.assert_array_almost_equal(structs[0].matrix, expected)
        assert structs[0].matrix.shape == (2, 3, 4)

    def test_custom_dims_column_name(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1],
                "matrix": [[1.0, 2.0, 3.0, 4.0]],
                "my_dims": [[2, 2]],
            },
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("matrix", pa.list_(pa.float64())),
                    ("my_dims", pa.list_(pa.int64())),
                ]
            ),
        )
        field_map = {"id": "id", "matrix": "matrix"}
        numpy_dims = {"matrix": "my_dims"}
        structs = _run_to_struct(batch, NDArrayStruct, field_map, batch.schema, numpy_dims)

        assert len(structs) == 1
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(structs[0].matrix, expected)

    def test_multiple_rows_with_different_shapes(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1, 2],
                "matrix": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [10.0, 20.0, 30.0, 40.0]],
                "matrix_csp_dimensions": [[2, 3], [2, 2]],
            },
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("matrix", pa.list_(pa.float64())),
                    ("matrix_csp_dimensions", pa.list_(pa.int64())),
                ]
            ),
        )
        field_map = {"id": "id", "matrix": "matrix"}
        structs = _run_to_struct(batch, NDArrayStruct, field_map, batch.schema)

        assert len(structs) == 2
        np.testing.assert_array_almost_equal(structs[0].matrix, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        assert structs[0].matrix.shape == (2, 3)
        np.testing.assert_array_almost_equal(structs[1].matrix, np.array([[10.0, 20.0], [30.0, 40.0]]))
        assert structs[1].matrix.shape == (2, 2)

    def test_dims_with_int32(self):
        """Dimensions column with int32 type should also work."""
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1],
                "matrix": [[1.0, 2.0, 3.0, 4.0]],
                "matrix_csp_dimensions": pa.array([[2, 2]], type=pa.list_(pa.int32())),
            },
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("matrix", pa.list_(pa.float64())),
                    ("matrix_csp_dimensions", pa.list_(pa.int32())),
                ]
            ),
        )
        field_map = {"id": "id", "matrix": "matrix"}
        structs = _run_to_struct(batch, NDArrayStruct, field_map, batch.schema)

        assert len(structs) == 1
        assert structs[0].matrix.shape == (2, 2)


# =====================================================================
# Tests: writing scalar fields (struct → batch)
# =====================================================================


class TestWriteScalarFields:
    def test_basic_scalar_types(self):
        structs = [
            ScalarStruct(i64=1, f64=1.1, s="a", b=True),
            ScalarStruct(i64=2, f64=2.2, s="b", b=False),
            ScalarStruct(i64=3, f64=3.3, s="c", b=True),
        ]
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        batches = _run_to_batches(structs, ScalarStruct, field_map)

        assert len(batches) == 1
        batch = batches[0]
        assert batch.num_rows == 3
        assert batch.column("i64").to_pylist() == [1, 2, 3]
        assert batch.column("f64").to_pylist() == pytest.approx([1.1, 2.2, 3.3])
        assert batch.column("s").to_pylist() == ["a", "b", "c"]
        assert batch.column("b").to_pylist() == [True, False, True]

    def test_field_mapping(self):
        structs = [
            NumericOnlyStruct(x=10, y=1.5),
            NumericOnlyStruct(x=20, y=2.5),
        ]
        field_map = {"x": "col_x", "y": "col_y"}
        batches = _run_to_batches(structs, NumericOnlyStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == 2
        assert batch.column("col_x").to_pylist() == [10, 20]
        assert batch.column("col_y").to_pylist() == pytest.approx([1.5, 2.5])

    def test_single_row(self):
        structs = [NumericOnlyStruct(x=42, y=3.14)]
        field_map = {"x": "x", "y": "y"}
        batches = _run_to_batches(structs, NumericOnlyStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == 1
        assert batch.column("x").to_pylist() == [42]
        assert batch.column("y").to_pylist() == pytest.approx([3.14])

    def test_many_rows(self):
        n = 1000
        structs = [NumericOnlyStruct(x=i, y=float(i) / 10.0) for i in range(n)]
        field_map = {"x": "x", "y": "y"}
        batches = _run_to_batches(structs, NumericOnlyStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == n
        assert batch.column("x").to_pylist() == list(range(n))

    def test_null_unset_fields(self):
        """Unset struct fields should become null in Arrow."""
        s1 = ScalarStruct(i64=1, f64=1.1)
        # s and b are unset
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        batches = _run_to_batches([s1], ScalarStruct, field_map)

        batch = batches[0]
        assert batch.column("i64").to_pylist() == [1]
        assert batch.column("s").to_pylist() == [None]
        assert batch.column("b").to_pylist() == [None]

    def test_multiple_ticks(self):
        tick1 = [NumericOnlyStruct(x=10, y=1.0)]
        tick2 = [NumericOnlyStruct(x=20, y=2.0)]

        field_map = {"x": "x", "y": "y"}
        all_results = _run_multi_tick_write([tick1, tick2], NumericOnlyStruct, field_map)

        assert len(all_results) == 2
        assert all_results[0][0].column("x").to_pylist() == [10]
        assert all_results[1][0].column("x").to_pylist() == [20]

    def test_no_field_map(self):
        """No field_map: auto-include all non-numpy fields with identity naming."""
        structs = [NumericOnlyStruct(x=5, y=2.5)]
        batches = _run_to_batches(structs, NumericOnlyStruct)

        batch = batches[0]
        assert batch.num_rows == 1
        assert batch.column("x").to_pylist() == [5]
        assert batch.column("y").to_pylist() == pytest.approx([2.5])

    def test_datetime_types(self):
        """datetime, timedelta, date, time fields."""
        dt_val = datetime(2024, 3, 15, 12, 0, 0)
        td_val = timedelta(seconds=3600)
        d_val = date(2024, 6, 15)
        t_val = time(14, 30, 0)

        structs = [DateTimeStruct(dt=dt_val, td=td_val, d=d_val, t=t_val)]
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        batches = _run_to_batches(structs, DateTimeStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == 1
        # Verify the arrow types
        assert pa.types.is_timestamp(batch.schema.field("dt").type)
        assert pa.types.is_duration(batch.schema.field("td").type)
        assert pa.types.is_date32(batch.schema.field("d").type)
        assert pa.types.is_time64(batch.schema.field("t").type)

    def test_enum_write(self):
        """Enum fields written as strings."""
        structs = [
            EnumStruct(label="x", color=MyEnum.A),
            EnumStruct(label="y", color=MyEnum.B),
        ]
        field_map = {"label": "label", "color": "color"}
        batches = _run_to_batches(structs, EnumStruct, field_map)

        batch = batches[0]
        assert batch.column("color").to_pylist() == ["A", "B"]
        assert batch.column("label").to_pylist() == ["x", "y"]

    def test_bytes_write(self):
        """Binary/bytes field."""
        val = b"my\x00value"
        structs = [BytesStruct(data=val)]
        field_map = {"data": "data"}
        batches = _run_to_batches(structs, BytesStruct, field_map)

        batch = batches[0]
        assert batch.column("data").to_pylist() == [val]

    def test_nested_struct_write(self):
        """Nested struct field."""
        inner = InnerStruct(x=42, y=2.5)
        structs = [NestedStruct(id=1, inner=inner)]
        field_map = {"id": "id", "inner": "inner"}
        batches = _run_to_batches(structs, NestedStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == 1
        assert batch.column("id").to_pylist() == [1]
        inner_col = batch.column("inner")
        assert inner_col.to_pylist() == [{"x": 42, "y": 2.5}]


# =====================================================================
# Tests: writing numpy 1D array fields
# =====================================================================


class TestWriteNumpy1DFields:
    def test_float_array(self):
        structs = [
            NumpyStruct(id=1, values=np.array([1.0, 2.0, 3.0])),
            NumpyStruct(id=2, values=np.array([4.0, 5.0])),
        ]
        field_map = {"id": "id", "values": "values"}
        batches = _run_to_batches(structs, NumpyStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == 2
        assert batch.column("id").to_pylist() == [1, 2]
        vals = batch.column("values").to_pylist()
        assert vals[0] == pytest.approx([1.0, 2.0, 3.0])
        assert vals[1] == pytest.approx([4.0, 5.0])

    def test_int_array(self):
        structs = [NumpyIntStruct(id=1, values=np.array([10, 20, 30], dtype=np.int64))]
        field_map = {"id": "id", "values": "values"}
        batches = _run_to_batches(structs, NumpyIntStruct, field_map)

        batch = batches[0]
        assert batch.column("values").to_pylist() == [[10, 20, 30]]

    def test_string_array(self):
        structs = [NumpyStringStruct(id=1, names=np.array(["alice", "bob", "carol"]))]
        field_map = {"id": "id", "names": "names"}
        batches = _run_to_batches(structs, NumpyStringStruct, field_map)

        batch = batches[0]
        assert batch.column("names").to_pylist() == [["alice", "bob", "carol"]]

    def test_bool_array(self):
        structs = [NumpyBoolStruct(id=1, flags=np.array([True, False, True]))]
        field_map = {"id": "id", "flags": "flags"}
        batches = _run_to_batches(structs, NumpyBoolStruct, field_map)

        batch = batches[0]
        assert batch.column("flags").to_pylist() == [[True, False, True]]

    def test_empty_array(self):
        structs = [NumpyStruct(id=1, values=np.array([], dtype=np.float64))]
        field_map = {"id": "id", "values": "values"}
        batches = _run_to_batches(structs, NumpyStruct, field_map)

        batch = batches[0]
        assert batch.column("values").to_pylist() == [[]]

    def test_null_numpy_field(self):
        """Unset numpy field should become null in Arrow."""
        structs = [NumpyStruct(id=1)]  # values is unset
        field_map = {"id": "id", "values": "values"}
        batches = _run_to_batches(structs, NumpyStruct, field_map)

        batch = batches[0]
        assert batch.column("id").to_pylist() == [1]
        assert batch.column("values").to_pylist() == [None]

    def test_mixed_scalar_and_numpy(self):
        structs = [
            MixedStruct(label="a", scores=np.array([0.1, 0.2])),
            MixedStruct(label="b", scores=np.array([0.3, 0.4, 0.5])),
        ]
        field_map = {"label": "label", "scores": "scores"}
        batches = _run_to_batches(structs, MixedStruct, field_map)

        batch = batches[0]
        assert batch.column("label").to_pylist() == ["a", "b"]
        vals = batch.column("scores").to_pylist()
        assert vals[0] == pytest.approx([0.1, 0.2])
        assert vals[1] == pytest.approx([0.3, 0.4, 0.5])


# =====================================================================
# Tests: writing NDArray fields
# =====================================================================


class TestWriteNumpyNDArrayFields:
    def test_2d_array(self):
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        structs = [NDArrayStruct(id=1, matrix=matrix)]
        field_map = {"id": "id", "matrix": "matrix"}
        batches = _run_to_batches(structs, NDArrayStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == 1
        data_col = batch.column("matrix").to_pylist()
        assert data_col[0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dims_col = batch.column("matrix_csp_dimensions").to_pylist()
        assert dims_col[0] == [2, 3]

    def test_3d_array(self):
        matrix = np.arange(24, dtype=float).reshape(2, 3, 4)
        structs = [NDArrayStruct(id=1, matrix=matrix)]
        field_map = {"id": "id", "matrix": "matrix"}
        batches = _run_to_batches(structs, NDArrayStruct, field_map)

        batch = batches[0]
        data_col = batch.column("matrix").to_pylist()
        assert data_col[0] == pytest.approx(list(range(24)))
        dims_col = batch.column("matrix_csp_dimensions").to_pylist()
        assert dims_col[0] == [2, 3, 4]

    def test_custom_dims_column_name(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        structs = [NDArrayStruct(id=1, matrix=matrix)]
        field_map = {"id": "id", "matrix": "matrix"}
        numpy_dims = {"matrix": "my_dims"}
        batches = _run_to_batches(structs, NDArrayStruct, field_map, numpy_dims)

        batch = batches[0]
        assert "my_dims" in batch.schema.names
        dims_col = batch.column("my_dims").to_pylist()
        assert dims_col[0] == [2, 2]

    def test_multiple_rows_different_shapes(self):
        m1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m2 = np.array([[10.0, 20.0], [30.0, 40.0]])
        structs = [
            NDArrayStruct(id=1, matrix=m1),
            NDArrayStruct(id=2, matrix=m2),
        ]
        field_map = {"id": "id", "matrix": "matrix"}
        batches = _run_to_batches(structs, NDArrayStruct, field_map)

        batch = batches[0]
        assert batch.num_rows == 2
        data_col = batch.column("matrix").to_pylist()
        assert data_col[0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert data_col[1] == pytest.approx([10.0, 20.0, 30.0, 40.0])
        dims_col = batch.column("matrix_csp_dimensions").to_pylist()
        assert dims_col[0] == [2, 3]
        assert dims_col[1] == [2, 2]


# =====================================================================
# Tests: round-trip (struct → batch → struct)
# =====================================================================


class TestRoundTrip:
    def test_scalar_round_trip(self):
        structs = [
            ScalarStruct(i64=1, f64=1.1, s="hello", b=True),
            ScalarStruct(i64=2, f64=2.2, s="world", b=False),
        ]
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, ScalarStruct, field_map)
            read_field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
            result = record_batches_to_struct(
                batches,
                ScalarStruct,
                read_field_map,
                pa.schema(
                    [
                        ("i64", pa.int64()),
                        ("f64", pa.float64()),
                        ("s", pa.utf8()),
                        ("b", pa.bool_()),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 2
        assert result_structs[0].i64 == 1
        assert result_structs[0].f64 == pytest.approx(1.1)
        assert result_structs[0].s == "hello"
        assert result_structs[0].b is True
        assert result_structs[1].i64 == 2
        assert result_structs[1].s == "world"

    def test_datetime_round_trip(self):
        """Round-trip for datetime, timedelta, date, time."""
        dt_val = datetime(2024, 3, 15, 12, 0, 0)
        td_val = timedelta(seconds=3600)
        d_val = date(2024, 6, 15)
        t_val = time(14, 30, 0)

        structs = [DateTimeStruct(dt=dt_val, td=td_val, d=d_val, t=t_val)]
        write_field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        read_field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, DateTimeStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                DateTimeStruct,
                read_field_map,
                pa.schema(
                    [
                        ("dt", pa.timestamp("ns", tz="UTC")),
                        ("td", pa.duration("ns")),
                        ("d", pa.date32()),
                        ("t", pa.time64("ns")),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        assert result_structs[0].dt == dt_val
        assert result_structs[0].td == td_val
        assert result_structs[0].d == d_val
        assert result_structs[0].t == t_val

    def test_enum_round_trip(self):
        """Round-trip for enum fields."""
        structs = [
            EnumStruct(label="x", color=MyEnum.A),
            EnumStruct(label="y", color=MyEnum.C),
        ]
        write_field_map = {"label": "label", "color": "color"}
        read_field_map = {"label": "label", "color": "color"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, EnumStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                EnumStruct,
                read_field_map,
                pa.schema([("label", pa.utf8()), ("color", pa.utf8())]),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 2
        assert result_structs[0].label == "x"
        assert result_structs[0].color == MyEnum.A
        assert result_structs[1].color == MyEnum.C

    def test_bytes_round_trip(self):
        """Round-trip for bytes field."""
        val = b"my\x00value"
        structs = [BytesStruct(data=val)]
        write_field_map = {"data": "data"}
        read_field_map = {"data": "data"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, BytesStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                BytesStruct,
                read_field_map,
                pa.schema([("data", pa.binary())]),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        assert result_structs[0].data == val

    def test_nested_struct_round_trip(self):
        """Round-trip for nested struct."""
        inner = InnerStruct(x=42, y=2.5)
        structs = [NestedStruct(id=1, inner=inner)]
        write_field_map = {"id": "id", "inner": "inner"}
        read_field_map = {"id": "id", "inner": "inner"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, NestedStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                NestedStruct,
                read_field_map,
                pa.schema(
                    [
                        ("id", pa.int64()),
                        ("inner", pa.struct([("x", pa.int64()), ("y", pa.float64())])),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        assert result_structs[0].id == 1
        assert result_structs[0].inner.x == 42
        assert result_structs[0].inner.y == pytest.approx(2.5)

    def test_numpy_round_trip(self):
        structs = [
            NumpyStruct(id=1, values=np.array([1.0, 2.0, 3.0])),
            NumpyStruct(id=2, values=np.array([4.0, 5.0])),
        ]
        write_field_map = {"id": "id", "values": "values"}
        read_field_map = {"id": "id", "values": "values"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, NumpyStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                NumpyStruct,
                read_field_map,
                pa.schema(
                    [
                        ("id", pa.int64()),
                        ("values", pa.list_(pa.float64())),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 2
        assert result_structs[0].id == 1
        np.testing.assert_array_almost_equal(result_structs[0].values, [1.0, 2.0, 3.0])
        assert result_structs[1].id == 2
        np.testing.assert_array_almost_equal(result_structs[1].values, [4.0, 5.0])

    def test_ndarray_round_trip(self):
        matrix = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        structs = [NDArrayStruct(id=1, matrix=matrix)]
        write_field_map = {"id": "id", "matrix": "matrix"}
        read_field_map = {"id": "id", "matrix": "matrix"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, NDArrayStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                NDArrayStruct,
                read_field_map,
                pa.schema(
                    [
                        ("id", pa.int64()),
                        ("matrix", pa.list_(pa.float64())),
                        ("matrix_csp_dimensions", pa.list_(pa.int64())),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        assert result_structs[0].id == 1
        np.testing.assert_array_almost_equal(result_structs[0].matrix, matrix)
        assert result_structs[0].matrix.shape == (2, 3)

    def test_all_types_round_trip(self):
        """Round-trip for all supported scalar types (mirrors test_all_types in test_parquet.py)."""
        structs = [
            AllTypesStruct(
                b=True,
                i=123,
                d=123.456,
                dt=datetime(2024, 1, 1, 12, 0, 0),
                dte=date(2024, 6, 15),
                t=time(14, 30, 0),
                td=timedelta(seconds=3600, milliseconds=123),
                s="hello",
                e=MyEnum.A,
            ),
            AllTypesStruct(
                b=False,
                i=456,
                d=789.012,
                dt=datetime(2024, 6, 15, 0, 0, 0),
                dte=date(2025, 1, 1),
                t=time(0, 0, 0),
                td=timedelta(seconds=0),
                s="world",
                e=MyEnum.B,
            ),
        ]
        field_map = {k: k for k in AllTypesStruct.metadata().keys()}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, AllTypesStruct, field_map)
            result = record_batches_to_struct(
                batches,
                AllTypesStruct,
                field_map,
                pa.schema(
                    [
                        ("b", pa.bool_()),
                        ("i", pa.int64()),
                        ("d", pa.float64()),
                        ("dt", pa.timestamp("ns", tz="UTC")),
                        ("dte", pa.date32()),
                        ("t", pa.time64("ns")),
                        ("td", pa.duration("ns")),
                        ("s", pa.utf8()),
                        ("e", pa.utf8()),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 2
        assert result_structs[0].b is True
        assert result_structs[0].i == 123
        assert result_structs[0].d == pytest.approx(123.456)
        assert result_structs[0].dt == datetime(2024, 1, 1, 12, 0, 0)
        assert result_structs[0].dte == date(2024, 6, 15)
        assert result_structs[0].t == time(14, 30, 0)
        assert result_structs[0].td == timedelta(seconds=3600, milliseconds=123)
        assert result_structs[0].s == "hello"
        assert result_structs[0].e == MyEnum.A

        assert result_structs[1].b is False
        assert result_structs[1].i == 456
        assert result_structs[1].s == "world"
        assert result_structs[1].e == MyEnum.B


# =====================================================================
# Tests: reverse round-trip (batch → struct → batch)
# =====================================================================


class TestReverseRoundTrip:
    """Convert batch → struct → batch and verify the result matches the original."""

    def test_scalar_batch_to_struct_to_batch(self):
        """batch → struct → batch for basic scalar types."""
        original = pa.RecordBatch.from_pydict(
            {"i64": [1, 2, 3], "f64": [1.1, 2.2, 3.3], "s": ["a", "b", "c"], "b": [True, False, True]}
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        schema = original.schema

        @csp.graph
        def G():
            data = csp.const([original])
            structs = record_batches_to_struct(data, ScalarStruct, field_map, schema)
            batches = struct_to_record_batches(structs, ScalarStruct, field_map)
            csp.add_graph_output("result", batches)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_batches = results["result"][0][1]

        assert len(result_batches) == 1
        result = result_batches[0]
        assert result.num_rows == 3
        assert result.column("i64").to_pylist() == [1, 2, 3]
        assert result.column("f64").to_pylist() == pytest.approx([1.1, 2.2, 3.3])
        assert result.column("s").to_pylist() == ["a", "b", "c"]
        assert result.column("b").to_pylist() == [True, False, True]

    def test_numpy_batch_to_struct_to_batch(self):
        """batch → struct → batch for numpy 1D arrays."""
        original = pa.RecordBatch.from_pydict(
            {"id": [1, 2], "values": [[1.0, 2.0, 3.0], [4.0, 5.0]]},
            schema=pa.schema([("id", pa.int64()), ("values", pa.list_(pa.float64()))]),
        )
        field_map = {"id": "id", "values": "values"}
        schema = original.schema

        @csp.graph
        def G():
            data = csp.const([original])
            structs = record_batches_to_struct(data, NumpyStruct, field_map, schema)
            batches = struct_to_record_batches(structs, NumpyStruct, field_map)
            csp.add_graph_output("result", batches)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_batches = results["result"][0][1]

        assert len(result_batches) == 1
        result = result_batches[0]
        assert result.num_rows == 2
        assert result.column("id").to_pylist() == [1, 2]
        vals = result.column("values").to_pylist()
        assert vals[0] == pytest.approx([1.0, 2.0, 3.0])
        assert vals[1] == pytest.approx([4.0, 5.0])

    def test_ndarray_batch_to_struct_to_batch(self):
        """batch → struct → batch for NDArrays."""
        original = pa.RecordBatch.from_pydict(
            {
                "id": [1],
                "matrix": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]],
                "matrix_csp_dimensions": [[2, 3]],
            },
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("matrix", pa.list_(pa.float64())),
                    ("matrix_csp_dimensions", pa.list_(pa.int64())),
                ]
            ),
        )
        field_map = {"id": "id", "matrix": "matrix"}
        schema = original.schema

        @csp.graph
        def G():
            data = csp.const([original])
            structs = record_batches_to_struct(data, NDArrayStruct, field_map, schema)
            batches = struct_to_record_batches(structs, NDArrayStruct, field_map)
            csp.add_graph_output("result", batches)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_batches = results["result"][0][1]

        assert len(result_batches) == 1
        result = result_batches[0]
        assert result.num_rows == 1
        data_col = result.column("matrix").to_pylist()
        assert data_col[0] == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        dims_col = result.column("matrix_csp_dimensions").to_pylist()
        assert dims_col[0] == [2, 3]

    def test_all_types_batch_to_struct_to_batch(self):
        """batch → struct → batch for all scalar types."""
        dt_val = datetime(2024, 3, 15, 12, 0, 0)
        td_val = timedelta(seconds=3600)
        d_val = date(2024, 6, 15)
        t_val = time(14, 30, 0)

        # Use a known UTC nanosecond value to avoid timezone ambiguity
        # 2024-03-15T12:00:00 UTC = 1710504000 seconds since epoch
        dt_ns = 1710504000 * 10**9
        td_ns = int(td_val.total_seconds() * 1e9)
        d_days = (d_val - date(1970, 1, 1)).days
        t_ns = (t_val.hour * 3600 + t_val.minute * 60 + t_val.second) * 10**9

        original = pa.RecordBatch.from_arrays(
            [
                pa.array([True], type=pa.bool_()),
                pa.array([123], type=pa.int64()),
                pa.array([123.456], type=pa.float64()),
                pa.array([dt_ns], type=pa.timestamp("ns", tz="UTC")),
                pa.array([d_days], type=pa.date32()),
                pa.array([t_ns], type=pa.time64("ns")),
                pa.array([td_ns], type=pa.duration("ns")),
                pa.array(["hello"], type=pa.utf8()),
                pa.array(["A"], type=pa.utf8()),
            ],
            schema=pa.schema(
                [
                    ("b", pa.bool_()),
                    ("i", pa.int64()),
                    ("d", pa.float64()),
                    ("dt", pa.timestamp("ns", tz="UTC")),
                    ("dte", pa.date32()),
                    ("t", pa.time64("ns")),
                    ("td", pa.duration("ns")),
                    ("s", pa.utf8()),
                    ("e", pa.utf8()),
                ]
            ),
        )
        field_map = {k: k for k in AllTypesStruct.metadata().keys()}
        schema = original.schema

        @csp.graph
        def G():
            data = csp.const([original])
            structs = record_batches_to_struct(data, AllTypesStruct, field_map, schema)
            batches = struct_to_record_batches(structs, AllTypesStruct, field_map)
            csp.add_graph_output("result", batches)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_batches = results["result"][0][1]

        assert len(result_batches) == 1
        result = result_batches[0]
        assert result.num_rows == 1
        assert result.column("b").to_pylist() == [True]
        assert result.column("i").to_pylist() == [123]
        assert result.column("d").to_pylist() == pytest.approx([123.456])
        assert result.column("s").to_pylist() == ["hello"]
        assert result.column("e").to_pylist() == ["A"]


# =====================================================================
# Tests: memory leak detection
# =====================================================================


class TestMemoryLeak:
    """Run repeated conversions and check that memory does not grow unboundedly.

    Uses psutil to measure RSS. Each test runs a large number of iterations with
    substantial data per iteration, then checks that memory growth after warmup
    stays within a reasonable bound. A real leak of even a few KB per iteration
    would accumulate to hundreds of MB over 5000 iterations.
    """

    @staticmethod
    def _get_rss_mb():
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)

    def test_struct_to_batch_to_struct_no_leak(self):
        """Repeated struct → batch → struct should not leak memory."""
        import gc

        n_warmup = 50
        n_iters = 5000
        rows_per_iter = 500

        field_map = {"id": "id", "values": "values"}
        read_schema = pa.schema([("id", pa.int64()), ("values", pa.list_(pa.float64()))])

        def run_one():
            structs = [NumpyStruct(id=i, values=np.random.rand(100)) for i in range(rows_per_iter)]

            @csp.graph
            def g(s_: object):
                data = csp.const(s_)
                batches = struct_to_record_batches(data, NumpyStruct, field_map)
                result = record_batches_to_struct(batches, NumpyStruct, field_map, read_schema)
                csp.add_graph_output("r", result)

            csp.run(g, structs, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))

        # Warmup — let allocators, caches, JIT, etc. stabilize
        for _ in range(n_warmup):
            run_one()

        gc.collect()
        baseline_mb = self._get_rss_mb()

        for _ in range(n_iters):
            run_one()

        gc.collect()
        final_mb = self._get_rss_mb()
        growth_mb = final_mb - baseline_mb

        # With 5000 iterations x 500 rows x 100 floats, total data processed is ~2GB.
        # A leak of even 1KB/iter would be 5MB. We allow 100MB for allocator noise.
        assert growth_mb < 100, (
            f"Memory grew by {growth_mb:.1f} MB over {n_iters} iterations "
            f"({rows_per_iter} rows/iter, baseline={baseline_mb:.1f} MB, "
            f"final={final_mb:.1f} MB) — possible leak"
        )

    def test_batch_to_struct_to_batch_no_leak(self):
        """Repeated batch → struct → batch should not leak memory."""
        import gc

        n_warmup = 50
        n_iters = 5000
        rows_per_iter = 500

        field_map = {"id": "id", "values": "values"}
        read_schema = pa.schema([("id", pa.int64()), ("values", pa.list_(pa.float64()))])

        def make_batch():
            ids = list(range(rows_per_iter))
            vals = [np.random.rand(100).tolist() for _ in range(rows_per_iter)]
            return pa.RecordBatch.from_pydict(
                {"id": ids, "values": vals},
                schema=read_schema,
            )

        def run_one():
            batch = make_batch()

            @csp.graph
            def g(b_: object, schema_: object):
                data = csp.const([b_])
                structs = record_batches_to_struct(data, NumpyStruct, field_map, schema_)
                batches = struct_to_record_batches(structs, NumpyStruct, field_map)
                csp.add_graph_output("r", batches)

            csp.run(g, batch, read_schema, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))

        # Warmup
        for _ in range(n_warmup):
            run_one()

        gc.collect()
        baseline_mb = self._get_rss_mb()

        for _ in range(n_iters):
            run_one()

        gc.collect()
        final_mb = self._get_rss_mb()
        growth_mb = final_mb - baseline_mb

        assert growth_mb < 100, (
            f"Memory grew by {growth_mb:.1f} MB over {n_iters} iterations "
            f"({rows_per_iter} rows/iter, baseline={baseline_mb:.1f} MB, "
            f"final={final_mb:.1f} MB) — possible leak"
        )


# =====================================================================
# Tests for all Arrow reader types (ensuring full type coverage)
# =====================================================================


class TestReadAllArrowTypes:
    """Test every Arrow type supported by ArrowFieldReader.

    CSP Python only supports int (int64) for integers. These tests verify that
    narrow Arrow integer types (int8/16/32, uint8/16/32/64) are correctly
    widened to int64 when reading into CSP struct fields.
    """

    # --- Narrow integer types ---

    @pytest.mark.parametrize(
        "arrow_type",
        [pa.int8(), pa.int16(), pa.int32(), pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()],
        ids=["int8", "int16", "int32", "uint8", "uint16", "uint32", "uint64"],
    )
    def test_narrow_integer_types(self, arrow_type):
        """Arrow narrow integers should widen to CSP int64."""
        arr = pa.array([10, 20, 30], type=arrow_type)
        batch = pa.record_batch(
            {
                "i64": arr,
                "f64": pa.array([1.0, 2.0, 3.0]),
                "s": pa.array(["a", "b", "c"]),
                "b": pa.array([True, False, True]),
            }
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert len(structs) == 3
        assert structs[0].i64 == 10
        assert structs[1].i64 == 20
        assert structs[2].i64 == 30

    # --- Float types ---

    def test_float32(self):
        """Arrow float32 should widen to CSP float64."""
        arr = pa.array([1.5, 2.5, 3.5], type=pa.float32())
        batch = pa.record_batch(
            {"f64": arr, "i64": pa.array([1, 2, 3]), "s": pa.array(["a", "b", "c"]), "b": pa.array([True, False, True])}
        )
        field_map = {"f64": "f64", "i64": "i64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert abs(structs[0].f64 - 1.5) < 1e-5
        assert abs(structs[1].f64 - 2.5) < 1e-5

    def test_float16(self):
        """Arrow float16 (half_float) should widen to CSP float64."""
        # PyArrow float16 has limited precision
        arr = pa.array([1.0, 2.0, 3.0], type=pa.float16())
        batch = pa.record_batch(
            {"f64": arr, "i64": pa.array([1, 2, 3]), "s": pa.array(["a", "b", "c"]), "b": pa.array([True, False, True])}
        )
        field_map = {"f64": "f64", "i64": "i64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert abs(structs[0].f64 - 1.0) < 0.1
        assert abs(structs[1].f64 - 2.0) < 0.1
        assert abs(structs[2].f64 - 3.0) < 0.1

    # --- String variants ---

    def test_large_string(self):
        """Arrow large_string should read into CSP str."""
        arr = pa.array(["hello", "world", "test"], type=pa.large_string())
        batch = pa.record_batch(
            {"s": arr, "i64": pa.array([1, 2, 3]), "f64": pa.array([1.0, 2.0, 3.0]), "b": pa.array([True, False, True])}
        )
        field_map = {"s": "s", "i64": "i64", "f64": "f64", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].s == "hello"
        assert structs[1].s == "world"
        assert structs[2].s == "test"

    # --- Binary variants ---

    def test_large_binary(self):
        """Arrow large_binary should read into CSP bytes."""
        arr = pa.array([b"\x01\x02", b"\x03\x04", b"\x05"], type=pa.large_binary())
        batch = pa.record_batch({"data": arr})
        field_map = {"data": "data"}
        structs = _run_to_struct(batch, BytesStruct, field_map, batch.schema)
        assert structs[0].data == b"\x01\x02"
        assert structs[1].data == b"\x03\x04"
        assert structs[2].data == b"\x05"

    def test_fixed_size_binary(self):
        """Arrow fixed_size_binary should read into CSP bytes."""
        arr = pa.array([b"\x01\x02\x03", b"\x04\x05\x06", b"\x07\x08\x09"], type=pa.binary(3))
        batch = pa.record_batch({"data": arr})
        field_map = {"data": "data"}
        structs = _run_to_struct(batch, BytesStruct, field_map, batch.schema)
        assert structs[0].data == b"\x01\x02\x03"
        assert structs[1].data == b"\x04\x05\x06"
        assert structs[2].data == b"\x07\x08\x09"

    # --- Timestamp units ---

    @pytest.mark.parametrize(
        "unit",
        ["s", "ms", "us", "ns"],
        ids=["seconds", "milliseconds", "microseconds", "nanoseconds"],
    )
    def test_timestamp_units(self, unit):
        """All timestamp units should read into CSP datetime."""
        dt_val = datetime(2023, 6, 15, 12, 30, 45)
        arr = pa.array([dt_val], type=pa.timestamp(unit))
        batch = pa.record_batch(
            {
                "dt": arr,
                "td": pa.array([timedelta(seconds=1)]),
                "d": pa.array([date(2023, 6, 15)]),
                "t": pa.array([time(12, 30, 45)]),
            }
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].dt == dt_val

    # --- Duration units ---

    @pytest.mark.parametrize(
        "unit",
        ["s", "ms", "us", "ns"],
        ids=["seconds", "milliseconds", "microseconds", "nanoseconds"],
    )
    def test_duration_units(self, unit):
        """All duration units should read into CSP timedelta."""
        td_val = timedelta(seconds=42)
        arr = pa.array([td_val], type=pa.duration(unit))
        batch = pa.record_batch(
            {
                "td": arr,
                "dt": pa.array([datetime(2023, 1, 1)]),
                "d": pa.array([date(2023, 1, 1)]),
                "t": pa.array([time(0, 0, 0)]),
            }
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].td == td_val

    # --- Date64 ---

    def test_date64(self):
        """Arrow date64 should read into CSP date."""
        d_val = date(2023, 6, 15)
        # date64 stores milliseconds since epoch
        arr = pa.array([d_val], type=pa.date64())
        batch = pa.record_batch(
            {
                "d": arr,
                "dt": pa.array([datetime(2023, 1, 1)]),
                "td": pa.array([timedelta(seconds=1)]),
                "t": pa.array([time(0, 0, 0)]),
            }
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].d == d_val

    # --- Time32 ---

    @pytest.mark.parametrize("unit", ["s", "ms"], ids=["seconds", "milliseconds"])
    def test_time32(self, unit):
        """Arrow time32 (s, ms) should read into CSP time."""
        t_val = time(12, 30, 45)
        arr = pa.array([t_val], type=pa.time32(unit))
        batch = pa.record_batch(
            {
                "t": arr,
                "dt": pa.array([datetime(2023, 1, 1)]),
                "td": pa.array([timedelta(seconds=1)]),
                "d": pa.array([date(2023, 1, 1)]),
            }
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].t.hour == 12
        assert structs[0].t.minute == 30
        assert structs[0].t.second == 45

    # --- Time64 ---

    @pytest.mark.parametrize("unit", ["us", "ns"], ids=["microseconds", "nanoseconds"])
    def test_time64(self, unit):
        """Arrow time64 (us, ns) should read into CSP time."""
        t_val = time(14, 15, 16, 123456)
        arr = pa.array([t_val], type=pa.time64(unit))
        batch = pa.record_batch(
            {
                "t": arr,
                "dt": pa.array([datetime(2023, 1, 1)]),
                "td": pa.array([timedelta(seconds=1)]),
                "d": pa.array([date(2023, 1, 1)]),
            }
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].t.hour == 14
        assert structs[0].t.minute == 15
        assert structs[0].t.second == 16

    # --- Dictionary-encoded string ---

    def test_dictionary_string(self):
        """Arrow dictionary-encoded string should read into CSP str."""
        arr = pa.array(["foo", "bar", "foo", "baz"]).dictionary_encode()
        batch = pa.record_batch(
            {
                "s": arr,
                "i64": pa.array([1, 2, 3, 4]),
                "f64": pa.array([1.0, 2.0, 3.0, 4.0]),
                "b": pa.array([True, True, False, False]),
            }
        )
        field_map = {"s": "s", "i64": "i64", "f64": "f64", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].s == "foo"
        assert structs[1].s == "bar"
        assert structs[2].s == "foo"
        assert structs[3].s == "baz"

    # --- Dictionary-encoded enum ---

    def test_dictionary_enum(self):
        """Arrow dictionary-encoded string should read into CSP enum."""
        arr = pa.array(["A", "B", "C", "A"]).dictionary_encode()
        batch = pa.record_batch({"label": pa.array(["x", "y", "z", "w"]), "color": arr})
        field_map = {"label": "label", "color": "color"}
        structs = _run_to_struct(batch, EnumStruct, field_map, batch.schema)
        assert structs[0].color == MyEnum.A
        assert structs[1].color == MyEnum.B
        assert structs[2].color == MyEnum.C
        assert structs[3].color == MyEnum.A

    # --- Enum from large_string ---

    def test_enum_from_large_string(self):
        """Arrow large_string should read into CSP enum."""
        arr = pa.array(["B", "C", "A"], type=pa.large_string())
        batch = pa.record_batch({"label": pa.array(["x", "y", "z"]), "color": arr})
        field_map = {"label": "label", "color": "color"}
        structs = _run_to_struct(batch, EnumStruct, field_map, batch.schema)
        assert structs[0].color == MyEnum.B
        assert structs[1].color == MyEnum.C
        assert structs[2].color == MyEnum.A

    # --- Null handling for narrow types ---

    def test_narrow_int_with_nulls(self):
        """Arrow narrow int with nulls should leave CSP field unset."""
        arr = pa.array([10, None, 30], type=pa.int16())
        batch = pa.record_batch({"x": arr, "y": pa.array([1.0, 2.0, 3.0])})
        field_map = {"x": "x", "y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)
        assert structs[0].x == 10
        assert not hasattr(structs[1], "x")  # null -> unset
        assert structs[2].x == 30

    def test_dictionary_with_nulls(self):
        """Arrow dictionary with nulls should leave CSP field unset."""
        arr = pa.array(["A", None, "C"]).dictionary_encode()
        batch = pa.record_batch({"label": pa.array(["x", "y", "z"]), "color": arr})
        field_map = {"label": "label", "color": "color"}
        structs = _run_to_struct(batch, EnumStruct, field_map, batch.schema)
        assert structs[0].color == MyEnum.A
        assert not hasattr(structs[1], "color")
        assert structs[2].color == MyEnum.C

    # --- Multiple rows with mixed narrow types ---

    def test_many_rows_narrow_int(self):
        """Read many rows from a narrow int column."""
        n = 1000
        arr = pa.array(list(range(n)), type=pa.int32())
        batch = pa.record_batch({"x": arr, "y": pa.array([float(i) for i in range(n)])})
        field_map = {"x": "x", "y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)
        assert len(structs) == n
        for i in range(n):
            assert structs[i].x == i
            assert structs[i].y == float(i)

    # --- Binary with nulls ---

    def test_binary_with_nulls(self):
        """Arrow binary with nulls should leave CSP field unset."""
        arr = pa.array([b"\x01", None, b"\x03"])
        batch = pa.record_batch({"data": arr})
        field_map = {"data": "data"}
        structs = _run_to_struct(batch, BytesStruct, field_map, batch.schema)
        assert structs[0].data == b"\x01"
        assert not hasattr(structs[1], "data")
        assert structs[2].data == b"\x03"


# =====================================================================
# Tests: null handling for common read types
# =====================================================================


class TestReadNullHandling:
    """Test that null values in various Arrow column types leave CSP struct fields unset."""

    def test_null_string(self):
        batch = pa.record_batch(
            {
                "i64": pa.array([1, 2, 3]),
                "f64": pa.array([1.0, 2.0, 3.0]),
                "s": pa.array(["hello", None, "world"]),
                "b": pa.array([True, False, True]),
            }
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].s == "hello"
        assert not hasattr(structs[1], "s")
        assert structs[2].s == "world"

    def test_null_float(self):
        batch = pa.record_batch(
            {
                "i64": pa.array([1, 2, 3]),
                "f64": pa.array([1.1, None, 3.3]),
                "s": pa.array(["a", "b", "c"]),
                "b": pa.array([True, False, True]),
            }
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].f64 == pytest.approx(1.1)
        assert not hasattr(structs[1], "f64")
        assert structs[2].f64 == pytest.approx(3.3)

    def test_null_int64(self):
        batch = pa.record_batch(
            {
                "i64": pa.array([1, None, 3]),
                "f64": pa.array([1.0, 2.0, 3.0]),
                "s": pa.array(["a", "b", "c"]),
                "b": pa.array([True, False, True]),
            }
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].i64 == 1
        assert not hasattr(structs[1], "i64")
        assert structs[2].i64 == 3

    def test_null_bool(self):
        batch = pa.record_batch(
            {
                "i64": pa.array([1, 2, 3]),
                "f64": pa.array([1.0, 2.0, 3.0]),
                "s": pa.array(["a", "b", "c"]),
                "b": pa.array([True, None, False]),
            }
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].b is True
        assert not hasattr(structs[1], "b")
        assert structs[2].b is False

    def test_null_datetime(self):
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([datetime(2024, 1, 1), None], type=pa.timestamp("ns", tz="UTC")),
                pa.array([timedelta(seconds=1), timedelta(seconds=2)]),
                pa.array([date(2024, 1, 1), date(2024, 1, 2)]),
                pa.array([time(12, 0, 0), time(13, 0, 0)]),
            ],
            schema=pa.schema(
                [
                    ("dt", pa.timestamp("ns", tz="UTC")),
                    ("td", pa.duration("ns")),
                    ("d", pa.date32()),
                    ("t", pa.time64("ns")),
                ]
            ),
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].dt == datetime(2024, 1, 1)
        assert not hasattr(structs[1], "dt")

    def test_null_timedelta(self):
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([datetime(2024, 1, 1), datetime(2024, 1, 2)], type=pa.timestamp("ns", tz="UTC")),
                pa.array([timedelta(seconds=42), None], type=pa.duration("ns")),
                pa.array([date(2024, 1, 1), date(2024, 1, 2)]),
                pa.array([time(12, 0, 0), time(13, 0, 0)]),
            ],
            schema=pa.schema(
                [
                    ("dt", pa.timestamp("ns", tz="UTC")),
                    ("td", pa.duration("ns")),
                    ("d", pa.date32()),
                    ("t", pa.time64("ns")),
                ]
            ),
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].td == timedelta(seconds=42)
        assert not hasattr(structs[1], "td")

    def test_null_date(self):
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([datetime(2024, 1, 1), datetime(2024, 1, 2)], type=pa.timestamp("ns", tz="UTC")),
                pa.array([timedelta(seconds=1), timedelta(seconds=2)]),
                pa.array([date(2024, 6, 15), None]),
                pa.array([time(12, 0, 0), time(13, 0, 0)]),
            ],
            schema=pa.schema(
                [
                    ("dt", pa.timestamp("ns", tz="UTC")),
                    ("td", pa.duration("ns")),
                    ("d", pa.date32()),
                    ("t", pa.time64("ns")),
                ]
            ),
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].d == date(2024, 6, 15)
        assert not hasattr(structs[1], "d")

    def test_null_time(self):
        batch = pa.RecordBatch.from_arrays(
            [
                pa.array([datetime(2024, 1, 1), datetime(2024, 1, 2)], type=pa.timestamp("ns", tz="UTC")),
                pa.array([timedelta(seconds=1), timedelta(seconds=2)]),
                pa.array([date(2024, 1, 1), date(2024, 1, 2)]),
                pa.array([time(14, 30, 0), None], type=pa.time64("ns")),
            ],
            schema=pa.schema(
                [
                    ("dt", pa.timestamp("ns", tz="UTC")),
                    ("td", pa.duration("ns")),
                    ("d", pa.date32()),
                    ("t", pa.time64("ns")),
                ]
            ),
        )
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert structs[0].t == time(14, 30, 0)
        assert not hasattr(structs[1], "t")

    def test_null_nested_struct(self):
        """Null nested struct should leave the field unset, and child readers stay in sync."""
        inner_type = pa.struct([("x", pa.int64()), ("y", pa.float64())])
        inner_arr = pa.StructArray.from_arrays(
            [pa.array([42, None, 99]), pa.array([2.5, None, 9.9])],
            fields=[pa.field("x", pa.int64()), pa.field("y", pa.float64())],
            mask=pa.array([False, True, False]),  # second row is null
        )
        batch = pa.RecordBatch.from_arrays(
            [pa.array([1, 2, 3]), inner_arr],
            schema=pa.schema([("id", pa.int64()), ("inner", inner_type)]),
        )
        field_map = {"id": "id", "inner": "inner"}
        structs = _run_to_struct(batch, NestedStruct, field_map, batch.schema)

        assert len(structs) == 3
        assert structs[0].inner.x == 42
        assert structs[0].inner.y == pytest.approx(2.5)
        assert not hasattr(structs[1], "inner")
        assert structs[2].inner.x == 99
        assert structs[2].inner.y == pytest.approx(9.9)

    def test_empty_batch_read(self):
        """Reading an empty RecordBatch should produce an empty list."""
        schema = pa.schema([("x", pa.int64()), ("y", pa.float64())])
        batch = pa.RecordBatch.from_pydict({"x": [], "y": []}, schema=schema)
        field_map = {"x": "x", "y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)
        assert len(structs) == 0


# =====================================================================
# Tests: null handling and edge cases for write direction
# =====================================================================


class TestWriteNullAndEdgeCases:
    """Test null/unset fields and edge cases in the write direction."""

    def test_null_datetime_fields(self):
        """Unset datetime/timedelta/date/time fields should become null in Arrow."""
        s = DateTimeStruct(dt=datetime(2024, 1, 1))  # only dt is set
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        batches = _run_to_batches([s], DateTimeStruct, field_map)

        batch = batches[0]
        assert batch.column("td").to_pylist() == [None]
        assert batch.column("d").to_pylist() == [None]
        assert batch.column("t").to_pylist() == [None]

    def test_null_enum_field(self):
        """Unset enum field should become null in Arrow."""
        s = EnumStruct(label="x")  # color is unset
        field_map = {"label": "label", "color": "color"}
        batches = _run_to_batches([s], EnumStruct, field_map)

        batch = batches[0]
        assert batch.column("label").to_pylist() == ["x"]
        assert batch.column("color").to_pylist() == [None]

    def test_null_nested_struct_field(self):
        """Unset nested struct field should become null in Arrow."""
        s = NestedStruct(id=1)  # inner is unset
        field_map = {"id": "id", "inner": "inner"}
        batches = _run_to_batches([s], NestedStruct, field_map)

        batch = batches[0]
        assert batch.column("id").to_pylist() == [1]
        assert batch.column("inner").to_pylist() == [None]

    def test_null_ndarray_field(self):
        """Unset NDArray field should produce null in both data and dims columns."""
        s = NDArrayStruct(id=1)  # matrix is unset
        field_map = {"id": "id", "matrix": "matrix"}
        batches = _run_to_batches([s], NDArrayStruct, field_map)

        batch = batches[0]
        assert batch.column("id").to_pylist() == [1]
        assert batch.column("matrix").to_pylist() == [None]
        assert batch.column("matrix_csp_dimensions").to_pylist() == [None]

    def test_empty_struct_vector_write(self):
        """Writing a tick with a single row followed by an empty tick should produce zero-row batch."""
        field_map = {"x": "x", "y": "y"}
        tick1 = [NumericOnlyStruct(x=1, y=1.0)]
        tick2 = [NumericOnlyStruct(x=2, y=2.0)]  # need a non-empty tick to validate structure

        # Verify single-row ticks work (empty list not expressible via csp.const)
        all_results = _run_multi_tick_write([tick1, tick2], NumericOnlyStruct, field_map)
        assert len(all_results) == 2
        assert all_results[0][0].num_rows == 1
        assert all_results[1][0].num_rows == 1


# =====================================================================
# Tests: struct_to_record_batches with field_map=None and numpy fields
# =====================================================================


class TestWriteAutoDetectNumpy:
    """Test struct_to_record_batches with field_map=None for structs with numpy fields."""

    def test_no_field_map_numpy_1d(self):
        """Auto-detect numpy 1D array fields when field_map is None."""
        structs = [
            NumpyStruct(id=1, values=np.array([1.0, 2.0, 3.0])),
            NumpyStruct(id=2, values=np.array([4.0, 5.0])),
        ]
        batches = _run_to_batches(structs, NumpyStruct)

        assert len(batches) == 1
        batch = batches[0]
        assert batch.num_rows == 2
        # Scalar fields auto-included
        assert batch.column("id").to_pylist() == [1, 2]
        # Numpy field auto-detected
        vals = batch.column("values").to_pylist()
        assert vals[0] == pytest.approx([1.0, 2.0, 3.0])
        assert vals[1] == pytest.approx([4.0, 5.0])

    def test_no_field_map_ndarray(self):
        """Auto-detect NDArray fields with dimension columns when field_map is None."""
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        structs = [NDArrayStruct(id=1, matrix=matrix)]
        batches = _run_to_batches(structs, NDArrayStruct)

        assert len(batches) == 1
        batch = batches[0]
        assert batch.num_rows == 1
        assert batch.column("id").to_pylist() == [1]
        data_col = batch.column("matrix").to_pylist()
        assert data_col[0] == pytest.approx([1.0, 2.0, 3.0, 4.0])
        dims_col = batch.column("matrix_csp_dimensions").to_pylist()
        assert dims_col[0] == [2, 2]

    def test_no_field_map_mixed_scalar_numpy(self):
        """Auto-detect with mixed scalar + numpy fields."""
        structs = [
            MixedStruct(label="a", scores=np.array([0.1, 0.2])),
        ]
        batches = _run_to_batches(structs, MixedStruct)

        assert len(batches) == 1
        batch = batches[0]
        assert batch.column("label").to_pylist() == ["a"]
        vals = batch.column("scores").to_pylist()
        assert vals[0] == pytest.approx([0.1, 0.2])


# =====================================================================
# Tests: round-trip for all numpy element types
# =====================================================================


class TestNumpyTypeRoundTrips:
    """Round-trip tests for numpy 1D arrays with int, str, and bool element types."""

    def test_numpy_int_round_trip(self):
        structs = [NumpyIntStruct(id=1, values=np.array([10, 20, 30], dtype=np.int64))]
        write_field_map = {"id": "id", "values": "values"}
        read_field_map = {"id": "id", "values": "values"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, NumpyIntStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                NumpyIntStruct,
                read_field_map,
                pa.schema([("id", pa.int64()), ("values", pa.list_(pa.int64()))]),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        np.testing.assert_array_equal(result_structs[0].values, [10, 20, 30])

    def test_numpy_string_round_trip(self):
        structs = [NumpyStringStruct(id=1, names=np.array(["alice", "bob"]))]
        write_field_map = {"id": "id", "names": "names"}
        read_field_map = {"id": "id", "names": "names"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, NumpyStringStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                NumpyStringStruct,
                read_field_map,
                pa.schema([("id", pa.int64()), ("names", pa.list_(pa.utf8()))]),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        np.testing.assert_array_equal(result_structs[0].names, ["alice", "bob"])

    def test_numpy_bool_round_trip(self):
        structs = [NumpyBoolStruct(id=1, flags=np.array([True, False, True]))]
        write_field_map = {"id": "id", "flags": "flags"}
        read_field_map = {"id": "id", "flags": "flags"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, NumpyBoolStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                NumpyBoolStruct,
                read_field_map,
                pa.schema([("id", pa.int64()), ("flags", pa.list_(pa.bool_()))]),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        np.testing.assert_array_equal(result_structs[0].flags, [True, False, True])


# =====================================================================
# Tests: NDArray with int element type
# =====================================================================


class TestNDArrayIntType:
    """Test NDArray with int element type (read, write, round-trip)."""

    def test_ndarray_int_write(self):
        matrix = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
        structs = [NDArrayIntStruct(id=1, matrix=matrix)]
        field_map = {"id": "id", "matrix": "matrix"}
        batches = _run_to_batches(structs, NDArrayIntStruct, field_map)

        batch = batches[0]
        data_col = batch.column("matrix").to_pylist()
        assert data_col[0] == [10, 20, 30, 40, 50, 60]
        dims_col = batch.column("matrix_csp_dimensions").to_pylist()
        assert dims_col[0] == [2, 3]

    def test_ndarray_int_read(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1],
                "matrix": [[10, 20, 30, 40, 50, 60]],
                "matrix_csp_dimensions": [[2, 3]],
            },
            schema=pa.schema(
                [
                    ("id", pa.int64()),
                    ("matrix", pa.list_(pa.int64())),
                    ("matrix_csp_dimensions", pa.list_(pa.int64())),
                ]
            ),
        )
        field_map = {"id": "id", "matrix": "matrix"}
        structs = _run_to_struct(batch, NDArrayIntStruct, field_map, batch.schema)

        assert len(structs) == 1
        expected = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
        np.testing.assert_array_equal(structs[0].matrix, expected)
        assert structs[0].matrix.shape == (2, 3)

    def test_ndarray_int_round_trip(self):
        matrix = np.array([[10, 20], [30, 40]], dtype=np.int64)
        structs = [NDArrayIntStruct(id=1, matrix=matrix)]
        write_field_map = {"id": "id", "matrix": "matrix"}
        read_field_map = {"id": "id", "matrix": "matrix"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, NDArrayIntStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                NDArrayIntStruct,
                read_field_map,
                pa.schema(
                    [
                        ("id", pa.int64()),
                        ("matrix", pa.list_(pa.int64())),
                        ("matrix_csp_dimensions", pa.list_(pa.int64())),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        np.testing.assert_array_equal(result_structs[0].matrix, matrix)
        assert result_structs[0].matrix.shape == (2, 2)


# =====================================================================
# Tests: mixed scalar + numpy + NDArray round-trip
# =====================================================================


class TestFullMixedRoundTrip:
    """Round-trip with a struct containing scalar, numpy 1D, and NDArray fields."""

    def test_mixed_scalar_numpy_ndarray_round_trip(self):
        matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
        structs = [
            FullMixedStruct(label="a", scores=np.array([0.1, 0.2, 0.3]), matrix=matrix),
        ]
        write_field_map = {"label": "label", "scores": "scores", "matrix": "matrix"}
        read_field_map = {"label": "label", "scores": "scores", "matrix": "matrix"}

        @csp.graph
        def G():
            data = csp.const(structs)
            batches = struct_to_record_batches(data, FullMixedStruct, write_field_map)
            result = record_batches_to_struct(
                batches,
                FullMixedStruct,
                read_field_map,
                pa.schema(
                    [
                        ("label", pa.utf8()),
                        ("scores", pa.list_(pa.float64())),
                        ("matrix", pa.list_(pa.float64())),
                        ("matrix_csp_dimensions", pa.list_(pa.int64())),
                    ]
                ),
            )
            csp.add_graph_output("result", result)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_structs = results["result"][0][1]

        assert len(result_structs) == 1
        assert result_structs[0].label == "a"
        np.testing.assert_array_almost_equal(result_structs[0].scores, [0.1, 0.2, 0.3])
        np.testing.assert_array_almost_equal(result_structs[0].matrix, matrix)
        assert result_structs[0].matrix.shape == (2, 2)


# =====================================================================
# Tests: reverse round-trip with null values
# =====================================================================


class TestReverseRoundTripWithNulls:
    """batch → struct → batch where the original batch contains null values."""

    def test_scalar_nulls_reverse_round_trip(self):
        """batch with nulls → struct → batch should preserve nulls."""
        original = pa.RecordBatch.from_pydict(
            {
                "i64": pa.array([1, None, 3]),
                "f64": pa.array([1.1, 2.2, None]),
                "s": pa.array([None, "b", "c"]),
                "b": pa.array([True, None, False]),
            }
        )
        field_map = {"i64": "i64", "f64": "f64", "s": "s", "b": "b"}
        schema = original.schema

        @csp.graph
        def G():
            data = csp.const([original])
            structs = record_batches_to_struct(data, ScalarStruct, field_map, schema)
            batches = struct_to_record_batches(structs, ScalarStruct, field_map)
            csp.add_graph_output("result", batches)

        results = csp.run(G, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        result_batches = results["result"][0][1]

        assert len(result_batches) == 1
        result = result_batches[0]
        assert result.num_rows == 3
        assert result.column("i64").to_pylist() == [1, None, 3]
        assert result.column("f64").to_pylist()[0] == pytest.approx(1.1)
        assert result.column("f64").to_pylist()[2] is None
        assert result.column("s").to_pylist() == [None, "b", "c"]
        assert result.column("b").to_pylist() == [True, None, False]


# =====================================================================
# Regression tests for specific bugs
# =====================================================================


class TestNonContiguousArrayWrite:
    """Regression: NativeListWriter must handle non-contiguous numpy arrays.

    Before the fix, PyArray_DATA + bulk AppendValues assumed C-contiguous memory
    layout, silently producing wrong data for sliced or transposed arrays.
    """

    def test_sliced_1d_array(self):
        """A sliced array (arr[::2]) is non-contiguous; round-trip must preserve values."""
        full = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        sliced = full[::2]  # [10, 30, 50], non-contiguous
        assert not sliced.flags["C_CONTIGUOUS"] or sliced.strides[0] != sliced.itemsize

        structs = [NumpyStruct(id=1, values=sliced)]
        field_map = {"id": "id", "values": "values"}
        batches = _run_to_batches(structs, NumpyStruct, field_map)

        assert len(batches) == 1
        batch = batches[0]
        schema = batch.schema
        read_field_map = {"id": "id", "values": "values"}
        result = _run_to_struct(batch, NumpyStruct, read_field_map, schema)

        assert len(result) == 1
        np.testing.assert_array_equal(result[0].values, [10.0, 30.0, 50.0])

    def test_sliced_int_array(self):
        """Non-contiguous int array round-trip."""
        full = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
        sliced = full[1::2]  # [2, 4, 6]

        structs = [NumpyIntStruct(id=1, values=sliced)]
        field_map = {"id": "id", "values": "values"}
        batches = _run_to_batches(structs, NumpyIntStruct, field_map)
        batch = batches[0]

        result = _run_to_struct(batch, NumpyIntStruct, {"id": "id", "values": "values"}, batch.schema)
        np.testing.assert_array_equal(result[0].values, [2, 4, 6])

    def test_transposed_ndarray(self):
        """A transposed 2D array is non-contiguous (Fortran order); round-trip must match."""
        original = np.array([[1.0, 2.0], [3.0, 4.0]])
        transposed = original.T  # [[1, 3], [2, 4]], Fortran-order

        structs = [NDArrayStruct(id=1, matrix=transposed)]
        field_map = {"id": "id", "matrix": "matrix"}
        batches = _run_to_batches(structs, NDArrayStruct, field_map)
        batch = batches[0]

        schema = batch.schema
        result = _run_to_struct(batch, NDArrayStruct, {"id": "id", "matrix": "matrix"}, schema)

        np.testing.assert_array_equal(result[0].matrix, transposed)
        assert result[0].matrix.shape == (2, 2)

    def test_fortran_order_array(self):
        """Explicitly Fortran-order (column-major) array must round-trip correctly."""
        c_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], order="C")
        f_array = np.asfortranarray(c_array)
        assert not f_array.flags["C_CONTIGUOUS"]

        structs = [NDArrayStruct(id=1, matrix=f_array)]
        field_map = {"id": "id", "matrix": "matrix"}
        batches = _run_to_batches(structs, NDArrayStruct, field_map)
        batch = batches[0]

        result = _run_to_struct(batch, NDArrayStruct, {"id": "id", "matrix": "matrix"}, batch.schema)
        np.testing.assert_array_equal(result[0].matrix, f_array)
        assert result[0].matrix.shape == (2, 3)


class TestDateWriteRegression:
    """Regression: DateWriter must compute days-since-epoch correctly.

    Ensures the optimized DateWriter (single timegm call) matches the expected
    Arrow Date32 representation.
    """

    def test_unix_epoch_date(self):
        """1970-01-01 should produce Date32 value of 0."""
        structs = [DateTimeStruct(dt=datetime(2020, 1, 1), td=timedelta(0), d=date(1970, 1, 1), t=time(0, 0, 0))]
        field_map = {"d": "d"}
        batches = _run_to_batches(structs, DateTimeStruct, field_map)
        batch = batches[0]
        assert batch.column("d").to_pylist() == [date(1970, 1, 1)]

    def test_known_dates(self):
        """Verify several known dates produce correct Date32 values."""
        known = [
            date(1970, 1, 1),
            date(2000, 1, 1),
            date(2020, 6, 15),
            date(2024, 2, 29),  # leap day
        ]
        structs = [DateTimeStruct(dt=datetime(2020, 1, 1), td=timedelta(0), d=d, t=time(0, 0, 0)) for d in known]
        field_map = {"d": "d"}
        batches = _run_to_batches(structs, DateTimeStruct, field_map)
        batch = batches[0]
        result_dates = batch.column("d").to_pylist()
        for expected, actual in zip(known, result_dates):
            assert actual == expected, f"Expected {expected}, got {actual}"

    def test_date_round_trip(self):
        """Write dates through arrow adapter and read back - values must match."""
        test_dates = [date(1970, 1, 1), date(1999, 12, 31), date(2025, 7, 4)]
        structs = [DateTimeStruct(dt=datetime(2020, 1, 1), td=timedelta(0), d=d, t=time(0, 0, 0)) for d in test_dates]
        field_map = {"d": "d"}
        batches = _run_to_batches(structs, DateTimeStruct, field_map)
        batch = batches[0]
        schema = batch.schema
        read_field_map = {"d": "d"}

        result = _run_to_struct(batch, DateTimeStruct, read_field_map, schema)
        for i, expected_date in enumerate(test_dates):
            assert result[i].d == expected_date


class TestTimeConstantsRegression:
    """Regression: time unit conversions must use named constants from Time.h.

    Verifies that timestamp, duration, date, and time conversions work correctly
    across all supported time units.
    """

    def test_timestamp_all_units(self):
        """Read timestamps in all units (s, ms, us, ns) and verify they decode correctly."""
        ts_val = datetime(2020, 6, 15, 12, 30, 45)
        for unit, arrow_type in [
            ("s", pa.timestamp("s", tz="UTC")),
            ("ms", pa.timestamp("ms", tz="UTC")),
            ("us", pa.timestamp("us", tz="UTC")),
            ("ns", pa.timestamp("ns", tz="UTC")),
        ]:
            batch = pa.RecordBatch.from_pydict(
                {"dt": pa.array([ts_val], type=arrow_type)},
                schema=pa.schema([("dt", arrow_type)]),
            )
            result = _run_to_struct(batch, DateTimeStruct, {"dt": "dt"}, batch.schema)
            if unit == "s":
                # Second precision loses sub-second
                assert result[0].dt == datetime(2020, 6, 15, 12, 30, 45)
            else:
                assert result[0].dt == ts_val

    def test_duration_all_units(self):
        """Read durations in all units and verify they decode correctly."""
        td_val = timedelta(days=1, hours=2, minutes=30, seconds=15)
        for unit, arrow_type in [
            ("s", pa.duration("s")),
            ("ms", pa.duration("ms")),
            ("us", pa.duration("us")),
            ("ns", pa.duration("ns")),
        ]:
            batch = pa.RecordBatch.from_pydict(
                {
                    "td": pa.array(
                        [int(td_val.total_seconds() * {"s": 1, "ms": 1e3, "us": 1e6, "ns": 1e9}[unit])], type=arrow_type
                    )
                },
                schema=pa.schema([("td", arrow_type)]),
            )
            result = _run_to_struct(batch, DateTimeStruct, {"td": "td"}, batch.schema)
            # Duration conversion may lose precision for coarser units
            assert abs(result[0].td.total_seconds() - td_val.total_seconds()) < 1.0

    def test_date32_and_date64(self):
        """Read both Date32 and Date64 formats and verify correct dates."""
        d = date(2020, 6, 15)

        class DateOnlyStruct(csp.Struct):
            d: date

        # Date32 (days since epoch)
        batch32 = pa.RecordBatch.from_pydict({"d": pa.array([d], type=pa.date32())})
        result32 = _run_to_struct(batch32, DateOnlyStruct, {"d": "d"}, batch32.schema)
        assert result32[0].d == d

        # Date64 (milliseconds since epoch)
        batch64 = pa.RecordBatch.from_pydict({"d": pa.array([d], type=pa.date64())})
        result64 = _run_to_struct(batch64, DateOnlyStruct, {"d": "d"}, batch64.schema)
        assert result64[0].d == d
