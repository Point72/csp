// Converts Arrow RecordBatches into csp::Struct instances.
//
// RecordBatchToStructConverter maps Arrow columns to CSP struct fields using
// FieldReader subclasses.  Scalar readers are auto-detected from the schema;
// additional readers (e.g. numpy array readers) can be injected at construction.

#ifndef _IN_CSP_ADAPTERS_ARROW_RecordBatchToStruct_H
#define _IN_CSP_ADAPTERS_ARROW_RecordBatchToStruct_H

#include <csp/adapters/arrow/ArrowFieldReader.h>
#include <csp/engine/Dictionary.h>
#include <arrow/record_batch.h>
#include <memory>
#include <string>
#include <vector>

namespace csp::adapters::arrow
{

class RecordBatchToStructConverter
{
public:
    // schema:         Arrow schema describing the RecordBatch columns
    // structMeta:     CSP StructMeta describing the target struct type
    // fieldMap:       optional column->field name mapping (null = match by column name)
    // customReaders:  additional readers for columns that need non-scalar handling (e.g. numpy arrays);
    //                 columns claimed by these readers are excluded from scalar auto-detection
    RecordBatchToStructConverter(
        const std::shared_ptr<::arrow::Schema> & schema,
        const std::shared_ptr<StructMeta> & structMeta,
        const DictionaryPtr & fieldMap = nullptr,
        std::vector<std::unique_ptr<FieldReader>> customReaders = {}
    );

    // Convert all rows from a RecordBatch into a vector of CSP Structs
    std::vector<StructPtr> convert( const ::arrow::RecordBatch & batch );

private:
    struct ScalarReaderEntry
    {
        std::unique_ptr<FieldReader> reader;
        int                          columnIndex;
    };

    std::shared_ptr<StructMeta>                m_structMeta;
    std::vector<ScalarReaderEntry>             m_scalarReaders;
    std::vector<std::unique_ptr<FieldReader>>  m_customReaders;
};

}

#endif
