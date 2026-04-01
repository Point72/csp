"""Tests for record_batches_to_struct and struct_to_record_batches.

Covers all scalar types supported by the parquet adapter (bool, int, float, str,
datetime, date, time, timedelta, enum, bytes, nested struct), numpy 1D arrays
(float, int, str, bool), NDArrays, field mapping, null handling, multiple ticks,
round-trips, and error cases.
"""

from datetime import date, datetime, time, timedelta

import pyarrow as pa
import pytest

import csp
from csp.adapters.arrow import record_batches_to_struct, struct_to_record_batches

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


# =====================================================================
# Helpers — reader direction (batch → struct)
# =====================================================================


def _run_to_struct(batches, cls, field_map, schema):
    """Run a graph that converts record batches to structs and returns the results."""

    @csp.graph
    def G(
        batches_: object,
        cls_: type,
        field_map_: dict,
        schema_: object,
    ):
        data = csp.const([batches_])
        structs = record_batches_to_struct(data, cls_, field_map_, schema_)
        csp.add_graph_output("structs", structs)

    results = csp.run(
        G,
        batches,
        cls,
        field_map,
        schema,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=1),
    )
    assert len(results["structs"]) == 1
    return results["structs"][0][1]


def _run_multi_tick_read(tick_batches, cls, field_map, schema):
    """Run a graph that ticks multiple lists of record batches and returns all results."""

    @csp.graph
    def G(
        ticks_: object,
        cls_: type,
        field_map_: dict,
        schema_: object,
    ):
        data = csp.unroll(csp.const(ticks_))
        structs = record_batches_to_struct(data, cls_, field_map_, schema_)
        csp.add_graph_output("structs", structs)

    results = csp.run(
        G,
        tick_batches,
        cls,
        field_map,
        schema,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=len(tick_batches)),
    )
    return [ts_val[1] for ts_val in results["structs"]]


# =====================================================================
# Helpers — writer direction (struct → batch)
# =====================================================================


def _run_to_batches(structs, cls, field_map=None):
    """Run a graph that converts structs to record batches and returns the results."""

    @csp.graph
    def G(
        structs_: object,
        cls_: type,
        field_map_: object,
    ):
        data = csp.const(structs_)
        batches = struct_to_record_batches(data, cls_, field_map_)
        csp.add_graph_output("batches", batches)

    results = csp.run(
        G,
        structs,
        cls,
        field_map,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=1),
    )
    assert len(results["batches"]) == 1
    return results["batches"][0][1]


def _run_multi_tick_write(tick_structs, cls, field_map=None):
    """Run a graph that ticks multiple lists of structs and returns all results."""

    @csp.graph
    def G(
        ticks_: object,
        cls_: type,
        field_map_: object,
    ):
        data = csp.unroll(csp.const(ticks_))
        batches = struct_to_record_batches(data, cls_, field_map_)
        csp.add_graph_output("batches", batches)

    results = csp.run(
        G,
        tick_structs,
        cls,
        field_map,
        starttime=_STARTTIME,
        endtime=_STARTTIME + timedelta(seconds=len(tick_structs)),
    )
    return [ts_val[1] for ts_val in results["batches"]]


# =====================================================================
# Helpers — round-trip (struct → batch → struct, and batch → struct → batch)
# =====================================================================


def _run_round_trip(structs, cls, field_map, schema):
    """struct → batch → struct round-trip; returns result structs."""

    @csp.graph
    def G(s_: object, cls_: type, fm_: object, schema_: object):
        data = csp.const(s_)
        batches = struct_to_record_batches(data, cls_, fm_)
        result = record_batches_to_struct(batches, cls_, fm_, schema_)
        csp.add_graph_output("result", result)

    results = csp.run(
        G, structs, cls, field_map, schema, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1)
    )
    return results["result"][0][1]


def _run_reverse_round_trip(batch, cls, field_map):
    """batch → struct → batch reverse round-trip; returns result batches."""
    schema = batch.schema

    @csp.graph
    def G(b_: object, cls_: type, fm_: dict, schema_: object):
        data = csp.const([b_])
        structs = record_batches_to_struct(data, cls_, fm_, schema_)
        batches = struct_to_record_batches(structs, cls_, fm_)
        csp.add_graph_output("result", batches)

    results = csp.run(G, batch, cls, field_map, schema, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
    return results["result"][0][1]


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
        field_map = {"x": "col_x", "y": "col_y"}
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

    # --- Null handling for all types ---

    def test_narrow_int_with_nulls(self):
        """Arrow narrow int with nulls should leave CSP field unset."""
        arr = pa.array([10, None, 30], type=pa.int16())
        batch = pa.record_batch({"x": arr, "y": pa.array([1.0, 2.0, 3.0])})
        field_map = {"x": "x", "y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)
        assert structs[0].x == 10
        assert not hasattr(structs[1], "x")  # null -> unset
        assert structs[2].x == 30

    def test_float16_with_nulls(self):
        """Arrow float16 with nulls should leave CSP field unset."""
        arr = pa.array([1.0, None, 3.0], type=pa.float16())
        batch = pa.record_batch(
            {"f64": arr, "i64": pa.array([1, 2, 3]), "s": pa.array(["a", "b", "c"]), "b": pa.array([True, False, True])}
        )
        field_map = {"f64": "f64", "i64": "i64", "s": "s", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert abs(structs[0].f64 - 1.0) < 0.1
        assert not hasattr(structs[1], "f64")
        assert abs(structs[2].f64 - 3.0) < 0.1

    def test_large_string_with_nulls(self):
        """Arrow large_string with nulls should leave CSP field unset."""
        arr = pa.array(["hello", None, "world"], type=pa.large_string())
        batch = pa.record_batch(
            {"s": arr, "i64": pa.array([1, 2, 3]), "f64": pa.array([1.0, 2.0, 3.0]), "b": pa.array([True, False, True])}
        )
        field_map = {"s": "s", "i64": "i64", "f64": "f64", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].s == "hello"
        assert not hasattr(structs[1], "s")
        assert structs[2].s == "world"

    def test_large_binary_with_nulls(self):
        """Arrow large_binary with nulls should leave CSP field unset."""
        arr = pa.array([b"\x01", None, b"\x03"], type=pa.large_binary())
        batch = pa.record_batch({"data": arr})
        field_map = {"data": "data"}
        structs = _run_to_struct(batch, BytesStruct, field_map, batch.schema)
        assert structs[0].data == b"\x01"
        assert not hasattr(structs[1], "data")
        assert structs[2].data == b"\x03"

    def test_fixed_size_binary_with_nulls(self):
        """Arrow fixed_size_binary with nulls should leave CSP field unset."""
        arr = pa.array([b"\x01\x02\x03", None, b"\x07\x08\x09"], type=pa.binary(3))
        batch = pa.record_batch({"data": arr})
        field_map = {"data": "data"}
        structs = _run_to_struct(batch, BytesStruct, field_map, batch.schema)
        assert structs[0].data == b"\x01\x02\x03"
        assert not hasattr(structs[1], "data")
        assert structs[2].data == b"\x07\x08\x09"

    def test_enum_from_string_with_nulls(self):
        """Arrow string enum with nulls should leave CSP field unset."""
        arr = pa.array(["A", None, "C"])
        batch = pa.record_batch({"label": pa.array(["x", "y", "z"]), "color": arr})
        field_map = {"label": "label", "color": "color"}
        structs = _run_to_struct(batch, EnumStruct, field_map, batch.schema)
        assert structs[0].color == MyEnum.A
        assert not hasattr(structs[1], "color")
        assert structs[2].color == MyEnum.C

    def test_enum_from_large_string_with_nulls(self):
        """Arrow large_string enum with nulls should leave CSP field unset."""
        arr = pa.array(["B", None, "A"], type=pa.large_string())
        batch = pa.record_batch({"label": pa.array(["x", "y", "z"]), "color": arr})
        field_map = {"label": "label", "color": "color"}
        structs = _run_to_struct(batch, EnumStruct, field_map, batch.schema)
        assert structs[0].color == MyEnum.B
        assert not hasattr(structs[1], "color")
        assert structs[2].color == MyEnum.A

    def test_dictionary_string_with_nulls(self):
        """Arrow dictionary-encoded string with nulls should leave CSP field unset."""
        arr = pa.array(["foo", None, "baz"]).dictionary_encode()
        batch = pa.record_batch(
            {"s": arr, "i64": pa.array([1, 2, 3]), "f64": pa.array([1.0, 2.0, 3.0]), "b": pa.array([True, False, True])}
        )
        field_map = {"s": "s", "i64": "i64", "f64": "f64", "b": "b"}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert structs[0].s == "foo"
        assert not hasattr(structs[1], "s")
        assert structs[2].s == "baz"

    def test_dictionary_enum_with_nulls(self):
        """Arrow dictionary-encoded enum with nulls should leave CSP field unset."""
        arr = pa.array(["A", None, "C"]).dictionary_encode()
        batch = pa.record_batch({"label": pa.array(["x", "y", "z"]), "color": arr})
        field_map = {"label": "label", "color": "color"}
        structs = _run_to_struct(batch, EnumStruct, field_map, batch.schema)
        assert structs[0].color == MyEnum.A
        assert not hasattr(structs[1], "color")
        assert structs[2].color == MyEnum.C

    def test_date64_with_nulls(self):
        """Arrow date64 with nulls should leave CSP field unset."""

        class DateOnlyStruct(csp.Struct):
            d: date

        arr = pa.array([date(2023, 6, 15), None, date(2024, 1, 1)], type=pa.date64())
        batch = pa.record_batch({"d": arr})
        field_map = {"d": "d"}
        structs = _run_to_struct(batch, DateOnlyStruct, field_map, batch.schema)
        assert structs[0].d == date(2023, 6, 15)
        assert not hasattr(structs[1], "d")
        assert structs[2].d == date(2024, 1, 1)

    def test_time32_with_nulls(self):
        """Arrow time32 with nulls should leave CSP field unset."""

        class TimeOnlyStruct(csp.Struct):
            t: time

        arr = pa.array([time(12, 30, 0), None, time(14, 0, 0)], type=pa.time32("s"))
        batch = pa.record_batch({"t": arr})
        field_map = {"t": "t"}
        structs = _run_to_struct(batch, TimeOnlyStruct, field_map, batch.schema)
        assert structs[0].t.hour == 12
        assert structs[0].t.minute == 30
        assert not hasattr(structs[1], "t")
        assert structs[2].t.hour == 14

    def test_time64_with_nulls(self):
        """Arrow time64 with nulls should leave CSP field unset."""

        class TimeOnlyStruct(csp.Struct):
            t: time

        arr = pa.array([time(14, 15, 16), None, time(0, 0, 1)], type=pa.time64("us"))
        batch = pa.record_batch({"t": arr})
        field_map = {"t": "t"}
        structs = _run_to_struct(batch, TimeOnlyStruct, field_map, batch.schema)
        assert structs[0].t.hour == 14
        assert not hasattr(structs[1], "t")
        assert structs[2].t.hour == 0
        assert structs[2].t.second == 1

    # --- All-null column ---

    def test_all_null_column(self):
        """A column where every value is null should leave all struct fields unset."""
        arr = pa.array([None, None, None], type=pa.int64())
        batch = pa.record_batch({"x": arr, "y": pa.array([1.0, 2.0, 3.0])})
        field_map = {"x": "x", "y": "y"}
        structs = _run_to_struct(batch, NumericOnlyStruct, field_map, batch.schema)
        assert len(structs) == 3
        for s in structs:
            assert not hasattr(s, "x")
        assert structs[0].y == pytest.approx(1.0)

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

    @pytest.mark.parametrize("null_field", ["i64", "f64", "s", "b"])
    def test_null_basic_scalar(self, null_field):
        """Null in a basic scalar column (int64, float64, string, bool) leaves the field unset."""
        data = {"i64": [1, 2, 3], "f64": [1.0, 2.0, 3.0], "s": ["a", "b", "c"], "b": [True, False, True]}
        data[null_field] = [data[null_field][0], None, data[null_field][2]]
        batch = pa.record_batch(data)
        field_map = {k: k for k in data}
        structs = _run_to_struct(batch, ScalarStruct, field_map, batch.schema)
        assert hasattr(structs[0], null_field)
        assert not hasattr(structs[1], null_field)
        assert hasattr(structs[2], null_field)

    @pytest.mark.parametrize("null_field", ["dt", "td", "d", "t"])
    def test_null_temporal(self, null_field):
        """Null in a temporal column (datetime, timedelta, date, time) leaves the field unset."""
        vals = {
            "dt": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "td": [timedelta(seconds=42), timedelta(seconds=2)],
            "d": [date(2024, 6, 15), date(2024, 1, 2)],
            "t": [time(14, 30, 0), time(13, 0, 0)],
        }
        # Set the null field's second value to None
        arrays = []
        schema_fields = [
            ("dt", pa.timestamp("ns", tz="UTC")),
            ("td", pa.duration("ns")),
            ("d", pa.date32()),
            ("t", pa.time64("ns")),
        ]
        for field_name, arrow_type in schema_fields:
            v = vals[field_name]
            if field_name == null_field:
                v = [v[0], None]
            arrays.append(pa.array(v, type=arrow_type))
        schema = pa.schema(schema_fields)
        batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
        field_map = {"dt": "dt", "td": "td", "d": "d", "t": "t"}
        structs = _run_to_struct(batch, DateTimeStruct, field_map, batch.schema)
        assert hasattr(structs[0], null_field)
        assert not hasattr(structs[1], null_field)

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
        result_batches = _run_reverse_round_trip(original, ScalarStruct, field_map)

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


# =====================================================================
# Struct definitions for multi-level nesting
# =====================================================================


class MiddleStruct(csp.Struct):
    tag: str
    inner: InnerStruct


class OuterStruct(csp.Struct):
    id: int
    middle: MiddleStruct
