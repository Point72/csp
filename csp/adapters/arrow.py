import itertools
import queue
import threading
from typing import Iterable, List, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import csp
from csp.impl.types.tstype import ts
from csp.impl.wiring import py_pull_adapter_def, py_push_adapter_def

__all__ = [
    "ArrowRealtimeAdapter",
    "ArrowHistoricalAdapter",
    "accumulate_record_batches",
]


class ArrowRealtimeAdapterImpl(csp.impl.pushadapter.PushInputAdapter):
    """Stream record batches in realtime into csp"""

    def __init__(self, timeout: int, source: queue.Queue[pa.RecordBatch]):
        """
        Args:
            timeout: max time in seconds to block for when waiting from results from the queue
            source: queue of streaming record batches, needs to be provided by the user
        """
        self.timeout = timeout
        self.queue = source
        self._thread = None
        self._running = False
        self._exc = None
        super().__init__()

    def start(self, start_time, end_time):
        self._thread = threading.Thread(target=self._run)
        self._running = True
        self._thread.start()

    def stop(self):
        if self._running:
            self._running = False
            self._thread.join()
            if self._exc:
                raise self._exc

    def _run(self):
        while self._running:
            try:
                new_batches = self.queue.get(block=True, timeout=self.timeout)
                self.push_tick(new_batches)
            except queue.Empty:
                # No new data loop back
                pass
            except Exception as e:
                self._exc = e
                break


ArrowRealtimeAdapter = py_push_adapter_def(
    "ArrowRealtimeAdapter",
    ArrowRealtimeAdapterImpl,
    ts[List[pa.RecordBatch]],
    timeout=int,
    source=queue.Queue[pa.RecordBatch],
)


class ArrowHistoricalAdapterImpl(csp.impl.pulladapter.PullInputAdapter):
    """Stream record batches from some source into csp"""

    def __init__(
        self,
        ts_col_name: str,
        stream: Optional[Iterable[pa.RecordBatch]],
        tables: Optional[Iterable[pa.Table]],
        filenames: Optional[Iterable[str]],
    ):
        """
        Args:
            ts_col_name: name of column that contains the timestamp field
            stream: an optional iterable of record batches
            tables: an optional iterable for arrow tables to read from
            filenames: an optional iterable of parquet files to read from

        NOTE: The user is responsible for ensuring that the data is sorted in ascending order on the 'ts_col_name' field
        NOTE: batches from stream, tables and filenames are iterated in that order
        """
        assert stream or filenames or tables, "Atleast one of stream, filenames, or tables must be not None"
        self.stream = stream
        self.tables = tables
        self.filenames = filenames
        self.ts_col_name = ts_col_name
        super().__init__()

    def start(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time

        # Info about the last chunk of data
        self.last_chunk = None
        self.last_ts = None
        # No of chunks in this batch
        self.batch_chunks_count = 0
        # Iterator for iterating over the chunks in a batch
        self.chunk_index_iter = None
        # No of chunks processed till now
        self.processed_chunks_count = 0
        # current batch being processed
        self.batch = None
        # all batches processed
        self.finished = False
        # start time filtering done
        self.filtered_start_time = False
        # the starting batch with start_time filtered
        self.starting_batch = None

        batch_iters = []
        if self.stream:
            batch_iters += [self.stream]

        if self.tables:
            batch_iters += [table.to_batches() for table in self.tables]

        if self.filenames:
            batch_iters += [pq.ParquetFile(filename).iter_batches() for filename in self.filenames]

        self.source = itertools.chain(*batch_iters)

        super().start(start_time, end_time)

    def next(self):
        if self.finished:
            return None

        # Filter out all batches which have ts < start time
        while not self.filtered_start_time and not self.finished:
            try:
                batch = next(self.source)
                if batch.num_rows != 0:
                    # NOTE: filter might be a good option to avoid this indirect way of computing the slice,
                    # however I am not sure if filter will be zero copy
                    valid_indices = pc.indices_nonzero(pc.greater_equal(batch[self.ts_col_name], self.start_time))
                    if len(valid_indices) != 0:
                        # Slice to only get the records with ts >= start_time
                        self.starting_batch = batch.slice(offset=valid_indices[0].as_py())
                        self.filtered_start_time = True
            except StopIteration:
                self.finished = True

        while not self.finished:
            # Process all the chunks in current batch
            if self.chunk_index_iter:
                try:
                    start_idx, next_start_idx = next(self.chunk_index_iter)
                    new_batches = [self.batch.slice(offset=start_idx, length=next_start_idx - start_idx)]
                    new_ts = self.batch[self.ts_col_name][start_idx].as_py()
                    self.processed_chunks_count += 1
                    if self.last_chunk:
                        if self.last_ts == new_ts:
                            new_batches = self.last_chunk + new_batches
                            self.last_chunk = None
                            self.last_ts = None
                        else:
                            raise Exception("last_chunk and new_batches have different timestamps")

                    if self.processed_chunks_count == self.batch_chunks_count:
                        self.last_chunk = new_batches
                        self.last_ts = new_ts
                        self.processed_chunks_count = 0
                    else:
                        if new_ts > self.end_time:
                            self.finished = True
                            continue
                        return (new_ts, new_batches)
                except StopIteration:
                    raise Exception("chunk_index_iter reached end, how?")

            # Try to get a new batch of data
            try:
                if self.starting_batch:
                    # Use the sliced batch from start_time filtering
                    self.batch = self.starting_batch
                    self.starting_batch = None
                else:
                    # Get the next batch of data
                    self.batch = next(self.source)
                    if self.batch.num_rows == 0:
                        continue

                all_timestamps = self.batch[self.ts_col_name]
                unique_timestamps = all_timestamps.unique()
                indexes = pc.index_in(unique_timestamps, all_timestamps).to_pylist() + [self.batch.num_rows]
                self.chunk_index_iter = zip(indexes, indexes[1:])
                self.batch_chunks_count = len(unique_timestamps)
                starting_ts = unique_timestamps[0].as_py()
                if starting_ts != self.last_ts and self.last_chunk:
                    new_batches = self.last_chunk
                    new_ts = self.last_ts
                    self.last_chunk = None
                    self.last_ts = None
                    if new_ts > self.end_time:
                        self.finished = True
                        continue
                    return (new_ts, new_batches)
            except StopIteration:
                self.finished = True
                if self.last_chunk:
                    if self.last_ts > self.end_time:
                        continue
                    return (self.last_ts, self.last_chunk)
        return None


ArrowHistoricalAdapter = py_pull_adapter_def(
    "ArrowHistoricalAdapter",
    ArrowHistoricalAdapterImpl,
    ts[List[pa.RecordBatch]],
    ts_col_name=str,
    stream=Optional[Iterable[pa.RecordBatch]],
    tables=Optional[Iterable[pa.Table]],
    filenames=Optional[Iterable[str]],
)


@csp.node
def accumulate_record_batches(filename: str, merge_record_batches: bool, batches: csp.ts[List[pa.RecordBatch]]):
    """
    Dump all the record batches to a parquet file

    Args:
        filename: name of file to write the data to
        merge_record_batches: A flag to combine all the record batches of a single tick into a single record batch (can save some space at the cost of memory)
        batches: The timeseries of list of record batches
    """
    with csp.state():
        s_writer = None
        s_filename = filename
        s_merge_batches = merge_record_batches

    with csp.stop():
        s_writer.close()

    if csp.ticked(batches):
        if s_merge_batches:
            batches = [pa.concat_batches(batches)]

        for batch in batches:
            if s_writer is None:
                s_writer = pq.ParquetWriter(s_filename, batch.schema)
            s_writer.write_batch(batch)
