"""
RecordBatch iterators for reading parquet/Arrow IPC/in-memory data.

These iterators yield (batch, basket_batches, maybe_schema_changed) tuples
that are consumed by the C++ adapter manager for type conversion and dispatch.

The protocol:
  - batch: pa.RecordBatch — one row group's worth of data
  - basket_batches: dict[str, pa.RecordBatch] — per-basket row data (empty dict if no baskets)
  - maybe_schema_changed: bool — True at file boundaries (first RG of each file)
"""

import os

import pyarrow as pa
import pyarrow.ipc
import pyarrow.parquet as pq


def parquet_file_rb_iterator(
    filenames_gen,
    starttime,
    endtime,
    needed_columns=None,
    allow_missing_files=False,
    allow_missing_columns=False,
    is_arrow_ipc=False,
):
    """Iterate RecordBatches from parquet (or Arrow IPC) files.

    :param filenames_gen: callable(starttime, endtime) -> iterable of filenames
    :param starttime: engine start time
    :param endtime: engine end time
    :param needed_columns: list of column names to read (None = all)
    :param allow_missing_files: skip files that don't exist
    :param allow_missing_columns: allow files missing some columns
    :param is_arrow_ipc: read Arrow IPC format instead of Parquet
    """
    for filename in filenames_gen(starttime, endtime):
        if not os.path.exists(filename):
            if allow_missing_files:
                continue
            raise FileNotFoundError(f"Parquet file not found: {filename}")

        if is_arrow_ipc:
            yield from _read_arrow_ipc_file(filename, needed_columns, allow_missing_columns)
        else:
            yield from _read_parquet_file(filename, needed_columns, allow_missing_columns)


def split_columns_rb_iterator(
    filenames_gen,
    starttime,
    endtime,
    needed_columns=None,
    allow_missing_columns=False,
    is_arrow_ipc=False,
):
    """Iterate RecordBatches from split-column parquet files.

    Each 'filename' from the generator is a directory. Each needed column
    is stored in a separate file: ``{directory}/{column_name}.parquet``
    (or ``.arrow`` for IPC).

    Row groups across column files are read in lockstep and merged into
    a single RecordBatch per row group.
    """
    ext = ".arrow" if is_arrow_ipc else ".parquet"
    for directory in filenames_gen(starttime, endtime):
        if not os.path.isdir(directory):
            raise NotADirectoryError(
                f"split_columns_to_files expects a directory, got: {directory}"
            )

        # Auto-discover columns from directory listing when needed_columns is None
        columns = needed_columns
        if columns is None:
            columns = []
            for f in sorted(os.listdir(directory)):
                if f.endswith(ext):
                    columns.append(f[: -len(ext)])

        # Identify dict baskets: columns ending in __csp_value_count
        basket_names = set()
        for col in columns:
            if col.endswith("__csp_value_count"):
                basket_names.add(col[: -len("__csp_value_count")])

        # Partition columns into main vs per-basket
        main_columns = []
        basket_columns = {}  # basket_name -> [col, ...]
        for col in columns:
            assigned = False
            for bname in basket_names:
                if col == bname or col == bname + "__csp_symbol" or col.startswith(bname + "."):
                    basket_columns.setdefault(bname, []).append(col)
                    assigned = True
                    break
            if not assigned:
                main_columns.append(col)

        # Open all column files lazily
        handles = {}
        for col in columns:
            col_path = os.path.join(directory, f"{col}{ext}")
            if not os.path.exists(col_path):
                if allow_missing_columns:
                    continue
                raise FileNotFoundError(f"Column file not found: {col_path}")
            if is_arrow_ipc:
                handles[col] = _open_ipc_file(col_path)
            else:
                handles[col] = pq.ParquetFile(col_path)

        if not handles:
            continue

        first_col = next(iter(handles))
        if is_arrow_ipc:
            readers = {col: h for col, h in handles.items()}
            while True:
                all_arrays = {}
                done = False
                for col, reader in readers.items():
                    try:
                        batch = reader.read_next_batch()
                        all_arrays[col] = batch.column(0)
                    except StopIteration:
                        done = True
                        break
                if done:
                    break
                main_batch = pa.RecordBatch.from_arrays(
                    [all_arrays[c] for c in main_columns if c in all_arrays],
                    names=[c for c in main_columns if c in all_arrays],
                )
                b_batches = {}
                for bname, bcols in basket_columns.items():
                    present = [c for c in bcols if c in all_arrays]
                    if present:
                        b_batches[bname] = pa.RecordBatch.from_arrays(
                            [all_arrays[c] for c in present], names=present,
                        )
                yield main_batch, b_batches, True
        else:
            num_rgs = handles[first_col].metadata.num_row_groups
            for rg_idx in range(num_rgs):
                all_arrays = {}
                all_fields = {}
                for col, pf in handles.items():
                    tbl = pf.read_row_group(rg_idx, columns=[col])
                    all_arrays[col] = tbl.column(0).chunk(0)
                    all_fields[col] = tbl.schema.field(0)

                main_schema = pa.schema([all_fields[c] for c in main_columns if c in all_arrays])
                main_batch = pa.RecordBatch.from_arrays(
                    [all_arrays[c] for c in main_columns if c in all_arrays],
                    schema=main_schema,
                )
                b_batches = {}
                for bname, bcols in basket_columns.items():
                    present = [c for c in bcols if c in all_arrays]
                    if present:
                        bschema = pa.schema([all_fields[c] for c in present])
                        b_batches[bname] = pa.RecordBatch.from_arrays(
                            [all_arrays[c] for c in present], schema=bschema,
                        )
                yield main_batch, b_batches, (rg_idx == 0)


def memory_table_rb_iterator(table_gen, starttime, endtime):
    """Iterate RecordBatches from in-memory Arrow Tables.

    :param table_gen: callable(starttime, endtime) -> iterable of pa.Table
    """
    for table in table_gen(starttime, endtime):
        if not isinstance(table, pa.Table):
            raise TypeError(f"Expected pyarrow.Table, got {type(table).__name__}")
        for i in range(table.column(0).num_chunks):
            # Extract one chunk as a RecordBatch
            arrays = []
            for col_idx in range(table.num_columns):
                arrays.append(table.column(col_idx).chunk(i))
            batch = pa.RecordBatch.from_arrays(arrays, schema=table.schema)
            yield batch, {}, (i == 0)

def _read_parquet_file(filename, needed_columns, allow_missing_columns):
    """Yield (batch, {}, maybe_schema_changed) from a single parquet file."""
    pf = pq.ParquetFile(filename)
    file_schema = pf.schema_arrow

    # Filter to columns that exist in the file
    columns_to_read = needed_columns
    if needed_columns is not None and allow_missing_columns:
        columns_to_read = [c for c in needed_columns if c in file_schema.names]

    first_batch = True
    for rg_idx in range(pf.metadata.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=columns_to_read)
        for batch in table.to_batches():
            yield batch, {}, first_batch
            first_batch = False


def _read_arrow_ipc_file(filename, needed_columns, allow_missing_columns):
    """Yield (batch, {}, maybe_schema_changed) from an Arrow IPC (streaming) file."""
    reader = pa.ipc.open_stream(filename)
    first_batch = True
    for batch in reader:
        if needed_columns is not None:
            if allow_missing_columns:
                available = set(batch.schema.names)
                batch = batch.select([c for c in needed_columns if c in available])
            else:
                try:
                    batch = batch.select(needed_columns)
                except KeyError:
                    missing = set(needed_columns) - set(batch.schema.names)
                    raise KeyError(f"Columns not found in IPC file: {missing}")
        yield batch, {}, first_batch
        first_batch = False


def _open_ipc_file(filename):
    """Open an Arrow IPC streaming file and return a RecordBatchReader."""
    return pa.ipc.open_stream(filename)
