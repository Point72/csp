#ifndef _IN_CSP_ADAPTERS_PARQUET_RecordBatchSink_H
#define _IN_CSP_ADAPTERS_PARQUET_RecordBatchSink_H

#include <arrow/record_batch.h>
#include <functional>
#include <memory>
#include <string>

namespace arrow { class Schema; }

namespace csp::adapters::parquet
{

// Callback interface for receiving RecordBatches from ParquetWriter.
// Python side implements this via a sink object.
struct RecordBatchSink
{
    using SchemaCallback     = std::function<void( const std::shared_ptr<::arrow::Schema> & )>;
    using BatchCallback      = std::function<void( const std::shared_ptr<::arrow::RecordBatch> & )>;
    using FileChangeCallback = std::function<void( const std::string & )>;
    using StopCallback       = std::function<void()>;

    SchemaCallback     onStart;       // called with schema at start()
    BatchCallback      onBatch;       // called with each full RecordBatch
    FileChangeCallback onFileChange;  // called when file rotation requested
    StopCallback       onStop;        // called at stop()
};

} // namespace csp::adapters::parquet

#endif
