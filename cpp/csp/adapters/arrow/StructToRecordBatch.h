// Converts csp::Struct instances into Arrow RecordBatches.
//
// StructToRecordBatchConverter maps CSP struct fields to Arrow columns using
// FieldWriter subclasses.  Scalar writers are auto-detected from the struct
// metadata; additional writers (e.g. numpy array writers) can be injected at
// construction.  Mirrors RecordBatchToStruct in the read direction.

#ifndef _IN_CSP_ADAPTERS_ARROW_StructToRecordBatch_H
#define _IN_CSP_ADAPTERS_ARROW_StructToRecordBatch_H

#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/engine/Dictionary.h>
#include <arrow/record_batch.h>
#include <memory>
#include <string>
#include <vector>

namespace csp::adapters::arrow
{

class StructToRecordBatchConverter
{
public:
    // structMeta:     CSP StructMeta describing the source struct type
    // fieldMap:       optional field->column name mapping (null = match by name, include all non-DIALECT_GENERIC fields)
    //                 when provided, only scalar fields listed in fieldMap are included
    // customWriters:  additional writers for fields that need non-scalar handling (e.g. numpy arrays)
    StructToRecordBatchConverter(
        const std::shared_ptr<StructMeta> & structMeta,
        const DictionaryPtr & fieldMap = nullptr,
        std::vector<std::unique_ptr<FieldWriter>> customWriters = {}
    );

    // Convert a vector of CSP Structs into one or more RecordBatches.
    // maxBatchSize controls the maximum number of rows per batch (0 = no limit).
    std::vector<std::shared_ptr<::arrow::RecordBatch>> convert( const std::vector<StructPtr> & structs,
                                                                int64_t maxBatchSize = 0 );

    // Get the Arrow schema for the output
    const std::shared_ptr<::arrow::Schema> & schema() const { return m_schema; }

private:
    std::shared_ptr<StructMeta>               m_structMeta;
    std::shared_ptr<::arrow::Schema>          m_schema;
    std::vector<std::unique_ptr<FieldWriter>> m_writers;
};

}

#endif
