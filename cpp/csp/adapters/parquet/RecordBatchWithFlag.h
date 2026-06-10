#ifndef _IN_CSP_ADAPTERS_PARQUET_RecordBatchWithFlag_H
#define _IN_CSP_ADAPTERS_PARQUET_RecordBatchWithFlag_H

#include <arrow/record_batch.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace csp::adapters::parquet
{

struct RecordBatchWithFlag
{
    std::shared_ptr<::arrow::RecordBatch> batch;
    std::unordered_map<std::string, std::shared_ptr<::arrow::RecordBatch>> basketBatches;
    bool                                  schemaChanged;
};

} // namespace csp::adapters::parquet

#endif
