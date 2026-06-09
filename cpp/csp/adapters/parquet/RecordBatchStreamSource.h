#ifndef _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H
#define _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H

#include <arrow/record_batch.h>
#include <csp/core/Time.h>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace csp::adapters::parquet
{

// A column data source providing schema (from file metadata) and batch iteration
// (from an async generator or sync reader).  Separates "what columns exist" from
// "how to read the next batch" so that async generators fit naturally without
// requiring a RecordBatchReader wrapper class.
//
// Lifecycle:
//   - Constructed by the stream source when a new file/stream is opened.
//   - Schema is derived from file metadata (no I/O).
//   - readNext wraps the underlying batch producer (async generator, sync reader, etc.)
//   - Destroying a ColumnSource while an async generator has in-flight futures is safe:
//     Arrow's generator state is ref-counted and any pending callbacks will complete
//     independently, releasing resources when done.
struct ColumnSource
{
    using ReadNextFn = std::function<::arrow::Status( std::shared_ptr<::arrow::RecordBatch> * )>;

    std::shared_ptr<::arrow::Schema>  schema;
    ReadNextFn                        readNext;
};

// Column name → source (schema + batch reader).
using ColumnSourceMap = std::map<std::string, std::shared_ptr<ColumnSource>>;

// Abstract interface for streaming RecordBatch data into C++.
// Implementations open files/streams and expose them as a flat dict of
// {column_name: ColumnSource}.
class RecordBatchStreamSource
{
public:
    using Ptr = std::shared_ptr<RecordBatchStreamSource>;

    virtual ~RecordBatchStreamSource() = default;

    // Initialize with time range and column projection.
    virtual void init( csp::DateTime start, csp::DateTime end,
                       const std::set<std::string> & neededColumns ) = 0;

    // Advance to the next stream set.
    virtual bool nextStream() = 0;

    // Column sources for the current stream.
    virtual const ColumnSourceMap & columnSources() const = 0;
};

} // namespace csp::adapters::parquet

#endif // _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H
