#ifndef _IN_CSP_ADAPTERS_PARQUET_ArrowBackedArrayBuilder_H
#define _IN_CSP_ADAPTERS_PARQUET_ArrowBackedArrayBuilder_H

#include <csp/adapters/parquet/ArrowSingleColumnArrayBuilder.h>

// Forward declare FieldWriter (defined in csp/adapters/arrow/ArrowFieldWriter.h) to keep this header light.
namespace csp::adapters::arrow { class FieldWriter; }

namespace csp::adapters::parquet
{

// Factory: create a scratch StructMeta + field for a given CSP type.
struct ScratchFieldInfo
{
    std::shared_ptr<StructMeta> meta;
    StructFieldPtr field;
};

ScratchFieldInfo createScratchField( const std::string & name, CspTypePtr cspType );

// Wraps an ArrowFieldWriter inside the ArrowSingleColumnArrayBuilder interface.
// Two modes:
//   Scratch mode: owns a single-field struct, caller writes value, handleRowFinished appends.
//   External mode: caller provides a Struct*, handleRowFinished reads field and appends.
class ArrowBackedArrayBuilder : public ArrowSingleColumnArrayBuilder
{
public:
    // Scratch mode: creates FieldWriter + scratch struct from a CspType.
    // Used by SingleColumnParquetOutputHandler.
    ArrowBackedArrayBuilder( const std::string & columnName, std::uint32_t chunkSize,
                             CspTypePtr cspType, bool isBytes = false );

    // External mode: creates FieldWriter from an existing StructFieldPtr.
    // Used by StructParquetOutputHandler (reads from source struct directly).
    ArrowBackedArrayBuilder( const std::string & columnName, std::uint32_t chunkSize,
                             const StructFieldPtr & structField );

    ~ArrowBackedArrayBuilder() override;

    // --- ArrowSingleColumnArrayBuilder interface ---
    std::shared_ptr<::arrow::DataType> getDataType() override;
    std::shared_ptr<::arrow::ArrayBuilder> getBuilder() override;
    int64_t length() const override;
    void handleRowFinished() override;
    std::shared_ptr<::arrow::Array> buildArray() override;

    // --- Scratch mode API ---
    // Get the scratch struct for value setting
    Struct * scratch() { return m_scratch.get(); }
    const StructFieldPtr & scratchField() const { return m_field; }

    // --- External mode API ---
    // Set the source struct for this row (marks value available)
    void setStruct( const Struct * s )
    {
        m_externalStruct    = s;
        m_hasExternalValue  = true;
    }

private:
    void init( const std::string & columnName, const StructFieldPtr & field );

    std::unique_ptr<csp::adapters::arrow::FieldWriter> m_writer;
    std::shared_ptr<StructMeta>                        m_scratchMeta; // non-null in scratch mode
    StructPtr                                          m_scratch;     // non-null in scratch mode
    StructFieldPtr                                     m_field;

    // External mode state
    const Struct * m_externalStruct   = nullptr;
    bool           m_hasExternalValue = false;
    bool           m_isScratchMode    = false;
};

// Factory: create ArrowBackedArrayBuilder for a given CspType (scratch mode).
std::shared_ptr<ArrowBackedArrayBuilder> createArrowBackedArrayBuilder(
    const std::string & columnName, std::uint32_t chunkSize,
    CspTypePtr cspType, bool isBytes = false );

// Factory: create ArrowBackedArrayBuilder for a struct field (external mode).
std::shared_ptr<ArrowBackedArrayBuilder> createArrowBackedArrayBuilderForField(
    const std::string & columnName, std::uint32_t chunkSize,
    const StructFieldPtr & structField );

} // namespace csp::adapters::parquet

#endif
