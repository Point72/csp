#ifndef _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H
#define _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H

#include <arrow/record_batch.h>
#include <csp/core/Time.h>
#include <map>
#include <memory>
#include <set>
#include <string>

namespace csp::adapters::parquet
{

// Column name → reader.
using ColumnReaderMap = std::map<std::string, std::shared_ptr<::arrow::RecordBatchReader>>;

// Abstract interface for streaming RecordBatch data into C++.
// Factories yield a flat dict of {column_name: RecordBatchReader}.
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

    // Column readers for the current stream.
    virtual const ColumnReaderMap & columnReaders() const = 0;
};

} // namespace csp::adapters::parquet

#endif // _IN_CSP_ADAPTERS_PARQUET_RecordBatchStreamSource_H
