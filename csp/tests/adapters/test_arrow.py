import math
import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy
import pandas
import polars
import pyarrow
import pyarrow as pa
import pyarrow.parquet
import pytest
import pytz

import csp
from csp.adapters.arrow import CRecordBatchPullInputAdapter, RecordBatchPullInputAdapter, write_record_batches
from csp.adapters.output_adapters.parquet import ParquetOutputConfig
from csp.adapters.parquet import ParquetReader, ParquetWriter

_STARTTIME = datetime(2020, 1, 1, 9, 0, 0)


@csp.graph
def G(ts_col_name: str, schema: pa.Schema, batches: object, expect_small: bool):
    data = RecordBatchPullInputAdapter(ts_col_name, batches, schema, expect_small)
    csp.add_graph_output("data", data)


class TestArrow:
    def make_record_batch(self, ts_col_name: str, row_size: int, ts: datetime) -> pa.RecordBatch:
        data = {
            ts_col_name: pa.array([ts] * row_size, type=pa.timestamp("s")),
            "name": pa.array([chr(ord("A") + idx % 26) for idx in range(row_size)]),
        }
        schema = pa.schema([(ts_col_name, pa.timestamp("s")), ("name", pa.string())])
        rb = pa.RecordBatch.from_pydict(data)
        return rb.cast(schema)

    def make_data(self, ts_col_name: str, row_sizes: [int], start: datetime = _STARTTIME, interval: int = 1):
        res = [
            self.make_record_batch(ts_col_name, row_size, start + timedelta(seconds=interval * idx))
            for idx, row_size in enumerate(row_sizes)
        ]
        return res[0].schema, res

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_ts_col_name(self, small_batches: bool):
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[1])
        with pytest.raises(KeyError):
            results = csp.run(G, "NotTsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_ts_col_type(self, small_batches: bool):
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[1])
        with pytest.raises(ValueError):
            results = csp.run(G, "name", rbs[0], rbs[1], small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_bad_source(self, small_batches: bool):
        rbs = (pa.schema([("TsCol", pa.timestamp("s"))]), 1)
        with pytest.raises(TypeError):
            results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME)

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_empty_rb(self, small_batches: bool):
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[0])
        results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[0, 0, 0])
        results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[0, 0, 0, 0, 0, 0])
        results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_start_not_found(self, small_batches: bool):
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[10])
        results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME + timedelta(days=1))
        assert len(results["data"]) == 0

        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[100, 10])
        results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME + timedelta(days=1))
        assert len(results["data"]) == 0

        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[100, 10, 1, 0, 0, 1, 2, 3, 4])
        results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME + timedelta(days=1))
        assert len(results["data"]) == 0

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_start_found(self, small_batches: bool):
        row_sizes = [10]
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        results = csp.run(G, "TsCol", rbs[0], rbs[1], small_batches, starttime=_STARTTIME + timedelta(days=-1))
        assert len(results["data"]) == 1
        assert results["data"][0][1] == rbs[1]

        row_sizes = [10, 11]
        rbs_prev = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes, start=_STARTTIME + timedelta(days=-1))
        rbs_new = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        rbs = rbs_prev[1] + rbs_new[1]
        results = csp.run(G, "TsCol", rbs_prev[0], rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 2
        for i in range(len(results["data"])):
            assert results["data"][i][1][0].equals(rbs_new[1][i])

        row_sizes = [10, 11]
        rbs_prev = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes, start=_STARTTIME + timedelta(days=-1))
        rbs_new = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        rbs = [pa.concat_batches(rbs_prev[1] + rbs_new[1])]
        results = csp.run(G, "TsCol", rbs_prev[0], rbs, small_batches, starttime=_STARTTIME)
        assert len(results["data"]) == 2
        for i in range(len(results["data"])):
            assert results["data"][i][1][0].equals(rbs_new[1][i])

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_split(self, small_batches: bool):
        row_sizes = [10]
        rbs_multi = [self.make_data(ts_col_name="TsCol", row_sizes=row_sizes) for i in range(10)]
        schema = rbs_multi[0][0]
        rbs = sum([rbs_multi[i][1] for i in range(len(rbs_multi))], [])
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        results = results["data"]
        assert len(results) == len(row_sizes)
        assert results[0][1] == rbs

        row_sizes = [10]
        rbs_multi = [self.make_data(ts_col_name="TsCol", row_sizes=row_sizes) for i in range(10)]
        schema = rbs_multi[0][0]
        rbs_1 = sum([rbs_multi[i][1] for i in range(len(rbs_multi))], [])
        rbs_multi = [
            self.make_data(ts_col_name="TsCol", row_sizes=row_sizes, start=_STARTTIME + timedelta(minutes=1))
            for i in range(10)
        ]
        rbs_2 = sum([rbs_multi[i][1] for i in range(len(rbs_multi))], [])
        rbs = rbs_1 + rbs_2
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        results = results["data"]
        assert len(results) == 2
        assert results[0][1] == rbs_1
        assert results[1][1] == rbs_2

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_different_sizes(self, small_batches: bool):
        row_sizes = [i for i in range(10)] + [0, 0, 0, 0] + [i for i in range(10)] + [0, 0, 0, 0]
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        schema = rbs[0]
        rbs = rbs[1]
        rbs_true = [rb for rb in rbs if len(rb)]
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        results = results["data"]
        assert len(results) == len(rbs_true)
        for i in range(len(rbs_true)):
            assert results[i][1][0] == rbs_true[i]

        row_sizes = [i for i in range(10)] + [0, 0, 0, 0] + [i for i in range(10)] + [0, 0, 0, 0]
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=row_sizes)
        schema = rbs[0]
        rbs_true = [rb for rb in rbs[1] if len(rb)]
        rbs = [pa.concat_batches(rbs[1])]
        results = csp.run(G, "TsCol", schema, rbs, small_batches, starttime=_STARTTIME)
        results = results["data"]
        assert len(results) == len(rbs_true)
        for i in range(len(rbs_true)):
            assert results[i][1][0] == rbs_true[i]

    @pytest.mark.parametrize("small_batches", (True, False))
    def test_end_time_early(self, small_batches: bool):
        rbs = self.make_data(ts_col_name="TsCol", row_sizes=[10])
        results = csp.run(
            G,
            "TsCol",
            rbs[0],
            rbs[1],
            small_batches,
            starttime=_STARTTIME - timedelta(days=1),
            endtime=_STARTTIME - timedelta(days=1) + timedelta(seconds=1),
        )
        assert len(results["data"]) == 0
