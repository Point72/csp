"""
RecordBatch stream factories for reading parquet/Arrow IPC/in-memory data.

These factories return callables that yield (main_reader, basket_readers)
tuples, where each reader is a pyarrow.RecordBatchReader backed by native
C++ Arrow readers (GIL-free after import via ArrowArrayStream).

The protocol:
  - main_reader: pa.RecordBatchReader — main column data
  - basket_readers: dict[str, pa.RecordBatchReader] — per-basket data (empty dict if no baskets)

Each tuple represents one file boundary. C++ imports the readers via
ArrowArrayStream and consumes batches natively without Python callbacks.
Schema changes are detected by comparing reader schemas at boundaries.
"""

import os

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc
import pyarrow.parquet as pq


def parquet_file_stream_factory(
    filenames_gen,
    allow_missing_files=False,
    is_arrow_ipc=False,
):
    """Create a stream factory for parquet (or Arrow IPC) files.

    Returns a callable(starttime, endtime, needed_columns) -> iterator of (reader, {}).
    C++ calls this at start() with the engine time range and needed columns.

    Column projection filters to columns available in each file.
    Missing column validation is handled by C++ setupFromSchema.
    """

    def factory(starttime, endtime, needed_columns):
        files = list(filenames_gen(starttime, endtime))
        if allow_missing_files:
            files = [f for f in files if os.path.exists(f)]
        else:
            for f in files:
                if not os.path.exists(f):
                    raise FileNotFoundError(f"Parquet file not found: {f}")

        if not files:
            return

        columns = list(needed_columns) if needed_columns else None

        for f in files:
            if is_arrow_ipc:
                reader = pa.ipc.open_stream(f)
                if columns:
                    table = reader.read_all()
                    available = [c for c in columns if c in table.column_names]
                    table = table.select(available)
                    yield (table.to_reader(), {})
                else:
                    yield (reader, {})
            else:
                file_ds = ds.dataset(f, format="parquet")
                proj = columns
                if columns:
                    proj = [c for c in columns if c in file_ds.schema.names]
                reader = file_ds.scanner(columns=proj or None).to_reader()
                yield (reader, {})

    return factory


def split_columns_stream_factory(
    filenames_gen,
    is_arrow_ipc=False,
):
    """Create a stream factory for split-column parquet files.

    Each 'filename' from the generator is a directory. Each column is stored
    in a separate file: ``{directory}/{column_name}.parquet`` (or ``.arrow``).

    Returns a callable(starttime, endtime, needed_columns) -> iterator of (main_reader, {basket: reader}).

    Missing column files are skipped (column projection filters to available).
    Missing column validation is handled by C++ setupFromSchema.
    """

    def factory(starttime, endtime, needed_columns):
        ext = ".arrow" if is_arrow_ipc else ".parquet"
        for directory in filenames_gen(starttime, endtime):
            if not os.path.isdir(directory):
                raise NotADirectoryError(f"split_columns_to_files expects a directory, got: {directory}")

            # Auto-discover columns from directory listing when needed_columns is None
            columns = list(needed_columns) if needed_columns else None
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

            # Read each column file into arrays; skip missing files
            all_arrays = {}
            all_fields = {}
            for col in columns:
                col_path = os.path.join(directory, f"{col}{ext}")
                if not os.path.exists(col_path):
                    continue
                if is_arrow_ipc:
                    reader = pa.ipc.open_stream(col_path)
                    batches = []
                    for batch in reader:
                        batches.append(batch.column(0))
                    all_arrays[col] = pa.concat_arrays(batches)
                    all_fields[col] = reader.schema.field(0)
                else:
                    tbl = pq.read_table(col_path, columns=[col])
                    all_arrays[col] = tbl.column(0).combine_chunks()
                    all_fields[col] = tbl.schema.field(0)

            if not all_arrays:
                continue

            # Build main table and reader
            present_main = [c for c in main_columns if c in all_arrays]
            if present_main:
                main_table = pa.table(
                    {c: all_arrays[c] for c in present_main},
                    schema=pa.schema([all_fields[c] for c in present_main]),
                )
            else:
                main_table = pa.table({})
            main_reader = main_table.to_reader()

            # Build per-basket tables and readers
            basket_readers = {}
            for bname, bcols in basket_columns.items():
                present = [c for c in bcols if c in all_arrays]
                if present:
                    btable = pa.table(
                        {c: all_arrays[c] for c in present},
                        schema=pa.schema([all_fields[c] for c in present]),
                    )
                    basket_readers[bname] = btable.to_reader()

            yield (main_reader, basket_readers)

    return factory


def memory_table_stream_factory(table_gen):
    """Create a stream factory for in-memory Arrow Tables.

    Returns a callable(starttime, endtime, needed_columns) -> iterator of (reader, {}).
    """

    def factory(starttime, endtime, needed_columns):
        for table in table_gen(starttime, endtime):
            if not isinstance(table, pa.Table):
                raise TypeError(f"Expected pyarrow.Table, got {type(table).__name__}")
            if needed_columns:
                available = set(table.column_names)
                cols = [c for c in needed_columns if c in available]
                table = table.select(cols)
            yield (table.to_reader(), {})

    return factory
