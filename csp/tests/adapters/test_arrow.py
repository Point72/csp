import tempfile
from datetime import datetime, timedelta

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import csp
from csp.adapters.arrow import RecordBatchPullInputAdapter, write_record_batches

_STARTTIME = datetime(2020, 1, 1, 9, 0, 0)


@csp.graph
def G(ts_col_name: str, schema: pa.Schema, batches: object, expect_small: bool):
    data = RecordBatchPullInputAdapter(ts_col_name, batches, schema, expect_small)
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
    def make_record_batch(self, ts_col_name: str, row_size: int, ts: datetime) -> pa.RecordBatch:
        data = {
            ts_col_name: pa.array([ts] * row_size, type=pa.timestamp("ms")),
            "name": pa.array([chr(ord("A") + idx % 26) for idx in range(row_size)]),
        }
        schema = pa.schema([(ts_col_name, pa.timestamp("ms")), ("name", pa.string())])
        return pa.RecordBatch.from_pydict(data, schema=schema)

    def make_data(self, ts_col_name: str, row_sizes: [int], start: datetime = _STARTTIME, interval: int = 1):
        res = [
            self.make_record_batch(ts_col_name, row_size, start + timedelta(seconds=interval * idx))
            for idx, row_size in enumerate(row_sizes)
        ]
        return res[0].schema, res, start, start + timedelta(seconds=interval * (len(row_sizes) - 1))

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_ts_col_name(self, small_batches: bool):
        schema, rbs, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=[1])
        with pytest.raises(KeyError):
            results = csp.run(G, "NotTsCol", schema, rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_ts_col_type(self, small_batches: bool):
        schema, rbs, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=[1])
        with pytest.raises(ValueError):
            results = csp.run(G, "name", schema, rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_source(self, small_batches: bool):
        schema, rbs = (pa.schema([("TsCol", pa.timestamp("s"))]), 1)
        with pytest.raises(TypeError):
            results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_empty_rb(self, small_batches: bool):
        schema, rbs, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=[0] * 1)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        schema, rbs, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=[0] * 3)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        schema, rbs, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=[0] * 4)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        schema, rbs, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=[0] * 1024)
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("row_sizes", ([10], [100, 10], [100, 10, 1, 0, 0, 1, 2, 3, 4]))
    @pytest.mark.parametrize("delta", (timedelta(microseconds=1), timedelta(seconds=1), timedelta(days=1)))
    def test_start_not_found(self, small_batches: bool, row_sizes: [int], delta: timedelta):
        schema, rbs, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=[10])
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=dt_start + delta)
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    @pytest.mark.parametrize("row_sizes", ([10], [100, 10], [1, 0, 2, 0, 3, 0]))
    @pytest.mark.parametrize("row_sizes_prev", ([10], [100, 10], [1, 0, 0, 1]))
    @pytest.mark.parametrize("delta", (timedelta(microseconds=1), timedelta(seconds=1), timedelta(days=1)))
    def test_start_found(self, small_batches: bool, row_sizes: [int], row_sizes_prev: [int], delta: timedelta):
        clean_row_sizes = [r for r in row_sizes if r != 0]
        schema, rbs_prev, _, old_dt_end = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        schema, rbs, dt_start, dt_end = self.make_data(
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
        schema, _, dt_start, dt_end = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        rbs_indivs = [[]] * dt_count
        rbs_full = []
        for idx in range(dt_count):
            _data = [
                self.make_data(ts_col_name="TsCol", row_sizes=row_sizes, start=dt_start + timedelta(seconds=idx))[1]
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
        schema, rbs, _, _ = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
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
        schema, rbs, _, _ = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
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
        _, rbs, _, _ = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
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
        _, rbs, _, _ = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
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
        _, rbs, _, _ = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
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
        _, rbs, _, _ = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        rbs_ts = [rbs]
        with tempfile.NamedTemporaryFile(prefix="csp_unit_tests", mode="w") as temp_file:
            temp_file.close()
            csp.run(WB, temp_file.name, False, 30, csp.unroll(csp.const(rbs_ts)), starttime=_STARTTIME)
            res = pq.read_table(temp_file.name)
            orig = pa.Table.from_batches(rbs)
            assert res.equals(orig)
            rbs_ts_expected = [_concat_batches(rbs[3 * i : 3 * i + 3]) for i in range(4)]
            assert rbs_ts_expected == res.to_batches()
