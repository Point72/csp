"""RecordBatch sink for parquet/arrow output.

Receives RecordBatches from C++ (via the Arrow C Data Interface)
and writes them using a pluggable writer backend.
"""

import os

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


# ── Writer helpers ────────────────────────────────────────────────────


class _IPCWriter:
    """Wraps an IPC stream writer + underlying file sink."""

    __slots__ = ("_writer", "_sink")

    def __init__(self, writer, sink):
        self._writer = writer
        self._sink = sink

    def write_batch(self, rb):
        self._writer.write_batch(rb)

    def close(self):
        self._writer.close()
        self._sink.close()


class _SplitColumnsWriter:
    """Dispatches each column of a RecordBatch to its own writer."""

    __slots__ = ("_writers", "_schema")

    def __init__(self, writers, schema):
        self._writers = writers
        self._schema = schema

    def write_batch(self, rb):
        for i, field in enumerate(self._schema):
            col_rb = pa.RecordBatch.from_arrays([rb.column(i)], schema=pa.schema([field]))
            self._writers[field.name].write_batch(col_rb)

    def close(self):
        for w in self._writers.values():
            w.close()


def _ensure_parent(path, allow_overwrite):
    if not allow_overwrite and os.path.exists(path):
        raise FileExistsError(f"File already exists: {path}")
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _apply_metadata(schema, file_metadata, column_metadata):
    """Attach file/column-level metadata to a pyarrow schema."""
    if file_metadata:
        existing = schema.metadata or {}
        merged = {**existing, **{k.encode(): v.encode() for k, v in file_metadata.items()}}
        schema = schema.with_metadata(merged)
    if column_metadata:
        fields = []
        for field in schema:
            if field.name in column_metadata:
                meta = {k.encode(): v.encode() for k, v in column_metadata[field.name].items()}
                existing = field.metadata or {}
                fields.append(field.with_metadata({**existing, **meta}))
            else:
                fields.append(field)
        schema = pa.schema(fields, metadata=schema.metadata)
    return schema


# ── Writer factories ──────────────────────────────────────────────────


def _parquet_writer_factory(compression, allow_overwrite, file_metadata=None, column_metadata=None):
    compression = compression or "none"

    def factory(path, schema):
        schema = _apply_metadata(schema, file_metadata, column_metadata)
        _ensure_parent(path, allow_overwrite)
        return pq.ParquetWriter(path, schema, compression=compression)

    return factory


def _arrow_ipc_writer_factory(compression, allow_overwrite):
    def factory(path, schema):
        _ensure_parent(path, allow_overwrite)
        options = ipc.IpcWriteOptions()
        if compression and compression != "none":
            options = ipc.IpcWriteOptions(compression=pa.Codec(compression))
        sink = pa.OSFile(path, "wb")
        return _IPCWriter(ipc.new_stream(sink, schema, options=options), sink)

    return factory


def _split_columns_writer_factory(compression, allow_overwrite, write_arrow_binary):
    compression = compression or "none"
    ext = ".arrow" if write_arrow_binary else ".parquet"

    def factory(dir_name, schema):
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        writers = {}
        for field in schema:
            col_schema = pa.schema([field])
            file_path = os.path.join(dir_name, field.name + ext)
            if not allow_overwrite and os.path.exists(file_path):
                raise FileExistsError(f"File already exists: {file_path}")
            if write_arrow_binary:
                sink = pa.OSFile(file_path, "wb")
                writers[field.name] = _IPCWriter(ipc.new_stream(sink, col_schema), sink)
            else:
                writers[field.name] = pq.ParquetWriter(file_path, col_schema, compression=compression)
        return _SplitColumnsWriter(writers, schema)

    return factory


# ── Unified sink ──────────────────────────────────────────────────────


class RecordBatchSink:
    """Receives RecordBatches from C++ and writes them via a pluggable backend.

    writer_factory(path, schema) must return an object with write_batch() and close().
    """

    def __init__(self, writer_factory, file_visitor=None):
        self._writer_factory = writer_factory
        self._file_visitor = file_visitor
        self._schema = None
        self._writer = None
        self._current_file = None

    def on_start(self, schema_capsule):
        self._schema = pa.Schema._import_from_c_capsule(schema_capsule)

    def on_batch(self, schema_capsule, array_capsule):
        rb = pa.RecordBatch._import_from_c_capsule(schema_capsule, array_capsule)
        if self._writer is not None:
            self._writer.write_batch(rb)

    def on_file_change(self, new_path):
        if self._writer is not None:
            self._writer.close()
            if self._file_visitor:
                self._file_visitor(self._current_file)
        if new_path:
            self._writer = self._writer_factory(new_path, self._schema)
            self._current_file = new_path
        else:
            self._writer = None

    def on_stop(self):
        if self._writer is not None:
            self._writer.close()
            if self._file_visitor:
                self._file_visitor(self._current_file)
            self._writer = None


# ── Public factory functions ──────────────────────────────────────────


def create_sink(file_name, compression, allow_overwrite, write_arrow_binary,
                split_columns_to_files, file_visitor=None,
                file_metadata=None, column_metadata=None):
    """Factory function to create the appropriate sink based on config."""
    if split_columns_to_files:
        wf = _split_columns_writer_factory(compression, allow_overwrite, write_arrow_binary)
    elif write_arrow_binary:
        wf = _arrow_ipc_writer_factory(compression, allow_overwrite)
    else:
        wf = _parquet_writer_factory(compression, allow_overwrite, file_metadata, column_metadata)
    return RecordBatchSink(wf, file_visitor)


def create_sink_factory(compression, allow_overwrite, write_arrow_binary, file_visitor=None):
    """Returns a callable(name) -> sink for dict basket writers.

    Dict basket writers always use split_columns_to_files=True for their data,
    and each basket gets its own sink identified by name.
    """
    def factory(name):
        wf = _split_columns_writer_factory(compression, allow_overwrite, write_arrow_binary)
        return RecordBatchSink(wf, file_visitor)
    return factory
