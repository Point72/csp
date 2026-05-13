import tempfile
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import csp
from csp.adapters.arrow import RecordBatchPullInputAdapter, struct_to_record_batches, write_record_batches

_STARTTIME = datetime(2020, 1, 1, 9, 0, 0)


def _make_record_batch(ts_col_name: str, row_size: int, ts: datetime) -> pa.RecordBatch:
    data = {
        ts_col_name: pa.array([ts] * row_size, type=pa.timestamp("ms")),
        "name": pa.array([chr(ord("A") + idx % 26) for idx in range(row_size)]),
    }
    schema = pa.schema([(ts_col_name, pa.timestamp("ms")), ("name", pa.string())])
    return pa.RecordBatch.from_pydict(data, schema=schema)


def _make_data(ts_col_name: str, row_sizes: list[int], start: datetime = _STARTTIME, interval: int = 1):
    res = [
        _make_record_batch(ts_col_name, row_size, start + timedelta(seconds=interval * idx))
        for idx, row_size in enumerate(row_sizes)
    ]
    return res, start, start + timedelta(seconds=interval * (len(row_sizes) - 1))


def _make_data_with_schema(ts_col_name: str, row_sizes: list[int], start: datetime = _STARTTIME, interval: int = 1):
    res, dt_start, dt_end = _make_data(ts_col_name, row_sizes, start, interval)
    return res[0].schema, res, dt_start, dt_end


@csp.graph
def G(ts_col_name: str, schema: pa.Schema, batches: object, expect_small: bool):
    data = RecordBatchPullInputAdapter(ts_col_name, batches, schema, expect_small_batches=expect_small)
    csp.add_graph_output("data", data)


@csp.graph
def G_lazy_schema(ts_col_name: str, batches: object, expect_small: bool):
    """Graph that passes schema=None for lazy schema extraction."""
    data = RecordBatchPullInputAdapter(ts_col_name, batches, schema=None, expect_small_batches=expect_small)
    csp.add_graph_output("data", data)


@csp.graph
def WB(where: str, merge: bool, batch_size: int, batches: csp.ts[[pa.RecordBatch]]):
    data = write_record_batches(where, batches, {}, merge, batch_size)


def _concat_batches(batches: list[pa.RecordBatch]) -> pa.RecordBatch:
    combined_table = pa.Table.from_batches(batches).combine_chunks()
    combined_batches = combined_table.to_batches()
    if len(combined_batches) > 1:
        raise ValueError("Not able to combine multiple record batches into one record batch")
    return combined_batches[0]


class TestArrow:
    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_ts_col_name(self, small_batches: bool):
        schema, rbs, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[1])
        with pytest.raises(KeyError):
            results = csp.run(G, "NotTsCol", schema, rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_ts_col_type(self, small_batches: bool):
        schema, rbs, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[1])
        with pytest.raises(ValueError):
            results = csp.run(G, "name", schema, rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_source(self, small_batches: bool):
        schema, rbs = (pa.schema([("TsCol", pa.timestamp("s"))]), 1)
        with pytest.raises(TypeError):
            results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_empty_rb(self, small_batches: bool):
        schema, rbs, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[0] * 1)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        schema, rbs, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[0] * 3)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        schema, rbs, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[0] * 4)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        schema, rbs, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[0] * 1024)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("row_sizes", ([10], [100, 10], [100, 10, 1, 0, 0, 1, 2, 3, 4]))
    @pytest.mark.parametrize("delta", (timedelta(microseconds=1), timedelta(seconds=1), timedelta(days=1)))
    def test_start_not_found(self, small_batches: bool, row_sizes: [int], delta: timedelta):
        schema, rbs, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[10])
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=dt_start + delta)
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("row_sizes", ([10], [100, 10], [1, 0, 2, 0, 3, 0]))
    @pytest.mark.parametrize("row_sizes_prev", ([10], [100, 10], [1, 0, 0, 1]))
    @pytest.mark.parametrize("delta", (timedelta(microseconds=1), timedelta(seconds=1), timedelta(days=1)))
    def test_start_found(self, small_batches: bool, row_sizes: [int], row_sizes_prev: [int], delta: timedelta):
        clean_row_sizes = [r for r in row_sizes if r != 0]
        schema, rbs_prev, _, old_dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        schema, rbs, dt_start, dt_end = _make_data_with_schema(
            ts_col_name="TsCol", row_sizes=row_sizes, start=old_dt_end + timedelta(days=10)
        )
        clean_rbs = [rb for rb in rbs if len(rb) != 0]
        full_rbs = rbs_prev + rbs
        results = csp.run(G, "TsCol", schema, full_rbs, small_batches, starttime=dt_start - delta)
        assert len(results["data"]) == len(clean_row_sizes)
        assert [len(r[1][0]) for r in results["data"]] == clean_row_sizes
        assert [r[1][0] for r in results["data"]] == clean_rbs

        results = csp.run(G, "TsCol", schema, [_concat_batches(full_rbs)], small_batches, starttime=dt_start - delta)
        assert len(results["data"]) == len(clean_row_sizes)
        assert [len(r[1][0]) for r in results["data"]] == clean_row_sizes
        assert [r[1][0] for r in results["data"]] == clean_rbs

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("row_sizes", ([10],))
    @pytest.mark.parametrize("repeat", (1, 10, 100))
    @pytest.mark.parametrize("dt_count", (1, 5))
    def test_split(self, small_batches: bool, row_sizes: [int], repeat: int, dt_count: int):
        schema, _, dt_start, dt_end = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        rbs_indivs = [[]] * dt_count
        rbs_full = []
        for idx in range(dt_count):
            _data = [
                _make_data_with_schema(
                    ts_col_name="TsCol", row_sizes=row_sizes, start=dt_start + timedelta(seconds=idx)
                )[1]
                for i in range(repeat)
            ]
            rbs_indivs[idx] = [item for sublist in _data for item in sublist]
            rbs_full += rbs_indivs[idx]
        results = csp.run(G, "TsCol", schema, rbs_full, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == len(rbs_indivs)
        assert [len(r[1]) for r in results["data"]] == [repeat] * dt_count
        for idx, tup in enumerate(results["data"]):
            assert tup[1] == rbs_indivs[idx]

        results = csp.run(G, "TsCol", schema, [_concat_batches(rbs_full)], small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == len(rbs_indivs)
        for idx, tup in enumerate(results["data"]):
            assert pa.Table.from_batches(tup[1]) == pa.Table.from_batches(rbs_indivs[idx])

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("row_sizes", ([10, 0, 0, 1], [0, 1, 0, 10]))
    def test_end_time_early(self, small_batches: bool, row_sizes: [int]):
        schema, rbs, _, _ = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        results = csp.run(
            G,
            "TsCol",
            schema,
            rbs,
            small_batches,
            starttime=_STARTTIME - timedelta(days=1),
            endtime=_STARTTIME - timedelta(days=1) + timedelta(seconds=1),
        )
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("seed", (1, 42, 100, 123))
    def test_different_size_rbs(self, small_batches: bool, seed: int):
        import random

        random.seed(seed)
        row_sizes = [random.randint(0, 100) for i in range(10000)]
        clean_row_sizes = [r for r in row_sizes if r != 0]
        schema, rbs, _, _ = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        clean_rbs = [rb for rb in rbs if len(rb) != 0]
        results = csp.run(
            G,
            "TsCol",
            schema,
            rbs,
            small_batches,
            starttime=_STARTTIME,
        )
        assert len(results["data"]) == len(clean_row_sizes)
        assert [len(r[1][0]) for r in results["data"]] == clean_row_sizes
        assert [r[1][0] for r in results["data"]] == clean_rbs

    @pytest.mark.parametrize("concat", (False, True))
    @pytest.mark.parametrize("row_sizes", ([1], [10], [1, 2, 3, 4, 5]))
    @pytest.mark.parametrize("batch_size", (1, 5, 10))
    def test_write_record_batches(self, row_sizes: [int], concat: bool, batch_size: int):
        _, rbs, _, _ = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        if not concat:
            rbs_ts = [[rb] for rb in rbs]
        else:
            rbs_ts = [rbs]
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            csp.run(WB, temp_file.name, concat, batch_size, csp.unroll(csp.const(rbs_ts)), starttime=_STARTTIME)
            res = pq.read_table(temp_file.name)
            orig = pa.Table.from_batches(rbs)
            assert res.equals(orig)

    @pytest.mark.parametrize("concat", (False, True))
    @pytest.mark.parametrize("row_sizes", ([1], [10], [1, 2, 3, 4, 5]))
    def test_write_record_batches_concat(self, row_sizes: [int], concat: bool):
        _, rbs, _, _ = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        if not concat:
            rbs_ts = [[rb] for rb in rbs]
        else:
            rbs_ts = [rbs]
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            csp.run(WB, temp_file.name, concat, 0, csp.unroll(csp.const(rbs_ts)), starttime=_STARTTIME)
            res = pq.read_table(temp_file.name)
            orig = pa.Table.from_batches(rbs)
            assert res.equals(orig)
            if not concat:
                rbs_ts_expected = [rb[0] for rb in rbs_ts]
            else:
                rbs_ts_expected = [_concat_batches(rbs_ts[0])]
            assert rbs_ts_expected == res.to_batches()

    def test_write_record_batches_batch_sizes(self):
        row_sizes = [10] * 10
        _, rbs, _, _ = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        rbs_ts = [rbs]
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            csp.run(WB, temp_file.name, False, 20, csp.unroll(csp.const(rbs_ts)), starttime=_STARTTIME)
            res = pq.read_table(temp_file.name)
            orig = pa.Table.from_batches(rbs)
            assert res.equals(orig)
            rbs_ts_expected = [_concat_batches(rbs[2 * i : 2 * i + 2]) for i in range(5)]
            assert rbs_ts_expected == res.to_batches()

        row_sizes = [10] * 10
        _, rbs, _, _ = _make_data_with_schema(ts_col_name="TsCol", row_sizes=row_sizes)
        rbs_ts = [rbs]
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            csp.run(WB, temp_file.name, False, 30, csp.unroll(csp.const(rbs_ts)), starttime=_STARTTIME)
            res = pq.read_table(temp_file.name)
            orig = pa.Table.from_batches(rbs)
            assert res.equals(orig)
            rbs_ts_expected = [_concat_batches(rbs[3 * i : 3 * i + 3]) for i in range(4)]
            assert rbs_ts_expected == res.to_batches()


class TestArrowLazySchema:
    """Tests for lazy schema initialization (schema=None).

    These tests verify that RecordBatchPullInputAdapter correctly extracts
    the schema from the first record batch when schema=None is passed.
    """

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_lazy_schema_basic(self, small_batches: bool):
        """Test basic lazy schema extraction from first batch."""
        rbs, _, _ = _make_data(ts_col_name="TsCol", row_sizes=[5])
        results = csp.run(G_lazy_schema, "TsCol", rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 1
        assert len(results["data"][0][1][0]) == 5

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_lazy_schema_multiple_batches(self, small_batches: bool):
        """Test lazy schema with multiple record batches."""
        rbs, _, _ = _make_data(ts_col_name="TsCol", row_sizes=[5, 10, 3])
        results = csp.run(G_lazy_schema, "TsCol", rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 3
        assert [len(r[1][0]) for r in results["data"]] == [5, 10, 3]

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_lazy_schema_empty_batches_before_data(self, small_batches: bool):
        """Test lazy schema extraction skips empty batches to find first non-empty."""
        rbs, _, _ = _make_data(ts_col_name="TsCol", row_sizes=[0, 0, 5, 10])
        results = csp.run(G_lazy_schema, "TsCol", rbs, small_batches, starttime=_STARTTIME)
        # Should get 2 results (the non-empty batches with rows 5 and 10)
        assert len(results["data"]) == 2
        assert [len(r[1][0]) for r in results["data"]] == [5, 10]

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_lazy_schema_all_empty(self, small_batches: bool):
        """Test lazy schema with all empty batches."""
        rbs, _, _ = _make_data(ts_col_name="TsCol", row_sizes=[0, 0, 0])
        results = csp.run(G_lazy_schema, "TsCol", rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_lazy_schema_bad_ts_col_name(self, small_batches: bool):
        """Test error handling for wrong timestamp column with lazy schema."""
        rbs, _, _ = _make_data(ts_col_name="TsCol", row_sizes=[5])
        with pytest.raises(ValueError):
            csp.run(G_lazy_schema, "NotTsCol", rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_lazy_schema_bad_ts_col_type(self, small_batches: bool):
        """Test error handling for non-timestamp column with lazy schema."""
        rbs, _, _ = _make_data(ts_col_name="TsCol", row_sizes=[5])
        with pytest.raises(ValueError):
            csp.run(G_lazy_schema, "name", rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("seed", (1, 42, 100))
    def test_lazy_schema_random_sizes(self, small_batches: bool, seed: int):
        """Test lazy schema with random sized batches."""
        import random

        random.seed(seed)
        row_sizes = [random.randint(0, 50) for i in range(100)]
        clean_row_sizes = [r for r in row_sizes if r != 0]
        rbs, _, _ = _make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        clean_rbs = [rb for rb in rbs if len(rb) != 0]
        results = csp.run(G_lazy_schema, "TsCol", rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == len(clean_row_sizes)
        assert [len(r[1][0]) for r in results["data"]] == clean_row_sizes
        assert [r[1][0] for r in results["data"]] == clean_rbs

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_lazy_schema_matches_explicit_schema(self, small_batches: bool):
        """Test that lazy schema produces same results as explicit schema."""
        schema, rbs, _, _ = _make_data_with_schema(ts_col_name="TsCol", row_sizes=[5, 0, 10, 3])

        # Run with explicit schema
        results_explicit = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)

        # Run with lazy schema
        results_lazy = csp.run(G_lazy_schema, "TsCol", rbs, small_batches, starttime=_STARTTIME)

        # Results should match
        assert len(results_explicit["data"]) == len(results_lazy["data"])
        for explicit, lazy in zip(results_explicit["data"], results_lazy["data"]):
            assert explicit[0] == lazy[0]  # timestamps match
            assert explicit[1] == lazy[1]  # data matches


class TestDeferredIterator:
    """Tests for deferred iterator behavior.

    These tests verify that the iterator is not consumed at graph build time,
    allowing lazy iterators to be set up after graph construction.
    """

    def test_deferred_iterator_not_consumed_at_build_time(self):
        """Test that iterator is not consumed during graph build, but is consumed at run time."""
        iteration_started = []

        def tracking_generator():
            iteration_started.append(True)
            yield _make_record_batch("TsCol", 5, _STARTTIME)

        gen = tracking_generator()

        @csp.graph
        def test_graph():
            data = RecordBatchPullInputAdapter("TsCol", gen, schema=None, expect_small_batches=False)
            csp.add_graph_output("data", data)

        # Graph definition should NOT consume the iterator
        assert len(iteration_started) == 0

        # Running the graph SHOULD consume it
        results = csp.run(test_graph, starttime=_STARTTIME)
        assert len(iteration_started) == 1
        assert len(results["data"]) == 1
        assert len(results["data"][0][1][0]) == 5

    def test_lazy_iterator_pattern(self):
        """Test the lazy iterator pattern used by LazyParquetIterator."""

        class LazyIterator:
            """Simulates LazyParquetIterator behavior."""

            def __init__(self):
                self._data = None

            def set_data(self, data):
                self._data = data

            def __iter__(self):
                if self._data is None:
                    raise RuntimeError("Data not set")
                for item in self._data:
                    yield item

        # Create lazy iterator without data
        lazy_iter = LazyIterator()

        # Create batches
        rbs = [
            _make_record_batch("TsCol", 5, _STARTTIME),
            _make_record_batch("TsCol", 3, _STARTTIME + timedelta(seconds=1)),
        ]

        # Set data before running (simulates what GraphComputeSimManager.start() does)
        lazy_iter.set_data(rbs)

        # Run with lazy schema
        results = csp.run(G_lazy_schema, "TsCol", lazy_iter, False, starttime=_STARTTIME)
        assert len(results["data"]) == 2
        assert [len(r[1][0]) for r in results["data"]] == [5, 3]


class TestStructConversion:
    """Tests for struct_to_record_batches conversion."""

    def test_nested_struct_with_nulls(self):
        """Regression test: NestedStructWriter::writeNull must not double-append to child builders.

        When a parent struct has a null nested struct field, the children of the
        struct column must still have exactly one entry per row.  A prior bug caused
        AppendNull() to also append empty values to children (on top of the explicit
        writeNull per child), shifting all subsequent valid values to wrong indices.
        """

        class Inner(csp.Struct):
            x: int
            y: float

        class Outer(csp.Struct):
            name: str
            inner: Inner

        @csp.node
        def validate_batches(batches: csp.ts[object]) -> csp.ts[object]:
            if csp.ticked(batches):
                return batches

        @csp.graph
        def g():
            s1 = Outer(name="a", inner=Inner(x=1, y=1.5))
            s2 = Outer(name="b")  # inner is unset -> null
            s3 = Outer(name="c", inner=Inner(x=3, y=3.5))
            s4 = Outer(name="d")  # inner is unset -> null
            s5 = Outer(name="e", inner=Inner(x=5, y=5.5))

            structs = csp.const([s1, s2, s3, s4, s5])
            batches = struct_to_record_batches(structs, Outer, max_batch_size=100)
            out = validate_batches(batches)
            csp.add_graph_output("batches", out)

        results = csp.run(g, starttime=_STARTTIME, endtime=_STARTTIME + timedelta(seconds=1))
        batch_list = results["batches"][0][1]
        batch = batch_list[0]

        assert batch.num_rows == 5

        struct_col = batch.column("inner")
        x_child = struct_col.field("x")
        y_child = struct_col.field("y")

        # Child arrays must have exactly the same length as the struct column
        assert len(x_child) == 5, f"child x length {len(x_child)} != struct length 5"
        assert len(y_child) == 5, f"child y length {len(y_child)} != struct length 5"

        # Verify actual values are correct (not shifted by double-append)
        values = struct_col.to_pylist()
        assert values[0] == {"x": 1, "y": 1.5}
        assert values[1] is None
        assert values[2] == {"x": 3, "y": 3.5}
        assert values[3] is None
        assert values[4] == {"x": 5, "y": 5.5}
