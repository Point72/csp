#ifndef _IN_CSP_ADAPTERS_PARQUET_ArrowBackedArrayBuilder_H
#define _IN_CSP_ADAPTERS_PARQUET_ArrowBackedArrayBuilder_H

#include <csp/adapters/parquet/ArrowSingleColumnArrayBuilder.h>

// Forward declare FieldWriter (defined in csp/adapters/arrow/ArrowFieldWriter.h) to keep this header light.
namespace csp::adapters::arrow { class FieldWriter; }

namespace csp::adapters::parquet
{

// Wraps an ArrowFieldWriter inside the ArrowSingleColumnArrayBuilder interface, in external mode:
// the caller provides the source Struct* for each row (setStruct), and handleRowFinished reads the
// field and appends (or appends null if no struct was set this row).
// Used by StructParquetOutputHandler. (Scalar columns use ScalarColumnArrayBuilder, which appends
// values directly with no struct.)
class ArrowBackedArrayBuilder : public ArrowSingleColumnArrayBuilder
{
public:
    ArrowBackedArrayBuilder( const std::string & columnName, std::uint32_t chunkSize,
                             const StructFieldPtr & structField );

    ~ArrowBackedArrayBuilder() override;

    // --- ArrowSingleColumnArrayBuilder interface ---
    std::shared_ptr<::arrow::DataType> getDataType() override;
    int64_t length() const override;
    void handleRowFinished() override;
    std::shared_ptr<::arrow::Array> buildArray() override;
    void reserve( int64_t numRows ) override;

    // Set the source struct for this row (marks value available)
    void setStruct( const Struct * s )
    {
        m_externalStruct    = s;
        m_hasExternalValue  = true;
    }

private:
    std::unique_ptr<csp::adapters::arrow::FieldWriter> m_writer;
    StructFieldPtr                                     m_field;

    const Struct * m_externalStruct   = nullptr;
    bool           m_hasExternalValue = false;
};

// Factory: create ArrowBackedArrayBuilder for a struct field (external mode).
std::shared_ptr<ArrowBackedArrayBuilder> createArrowBackedArrayBuilderForField(
    const std::string & columnName, std::uint32_t chunkSize,
    const StructFieldPtr & structField );

} // namespace csp::adapters::parquet

#endif
