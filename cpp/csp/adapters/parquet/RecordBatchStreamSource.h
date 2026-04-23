#ifndef _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H
#define _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H

#include <arrow/record_batch.h>
#include <csp/core/Time.h>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

namespace csp::adapters::parquet
{

// Abstract interface for streaming RecordBatch data from Python.
// The Python implementation wraps a "stream factory" callable that yields
// (RecordBatchReader, {basket: RecordBatchReader}) tuples.
// C++ imports readers via ArrowArrayStream and pulls batches natively.
class RecordBatchStreamSource
{
public:
    using Ptr = std::shared_ptr<RecordBatchStreamSource>;

    virtual ~RecordBatchStreamSource() = default;

    // Call the stream factory with (starttime, endtime, needed_columns).
    // Must be called before nextStream().
    virtual void init( csp::DateTime start, csp::DateTime end,
                       const std::set<std::string> & neededColumns ) = 0;

    // Advance to the next stream set (file/directory boundary).
    // After this call, mainReader() and basketBatches() reflect the new data.
    // Returns false when all data is exhausted.
    virtual bool nextStream() = 0;

    // The current main RecordBatchReader. Pull batches with ReadNext().
    // Returns nullptr before the first nextStream() call.
    virtual std::shared_ptr<::arrow::RecordBatchReader> mainReader() = 0;

    // Pre-read basket batches for the current stream boundary.
    // Empty when there are no dict baskets.
    virtual const std::unordered_map<std::string, std::shared_ptr<::arrow::RecordBatch>> & basketBatches() const = 0;
};

} // namespace csp::adapters::parquet

#endif
