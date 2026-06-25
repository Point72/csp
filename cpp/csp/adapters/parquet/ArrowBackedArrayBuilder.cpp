#include <csp/adapters/parquet/ArrowBackedArrayBuilder.h>
#include <csp/adapters/arrow/ArrowFieldWriter.h>

namespace csp::adapters::parquet
{

ArrowBackedArrayBuilder::ArrowBackedArrayBuilder(
    const std::string & columnName, std::uint32_t chunkSize,
    const StructFieldPtr & structField )
    : ArrowSingleColumnArrayBuilder( columnName, chunkSize )
    , m_field( structField )
{
    auto created = csp::adapters::arrow::createFieldWriter( columnName, m_field );
    m_writer = std::move( created.writer );
    // Reserve the first chunk up front so the per-row append path doesn't reallocate as rows accumulate.
    m_writer -> reserve( getChunkSize() );
}

ArrowBackedArrayBuilder::~ArrowBackedArrayBuilder() = default;

void ArrowBackedArrayBuilder::reserve( int64_t numRows )
{
    m_writer -> reserve( numRows );
}

std::shared_ptr<::arrow::DataType> ArrowBackedArrayBuilder::getDataType()
{
    return m_writer -> dataTypes()[0];
}

int64_t ArrowBackedArrayBuilder::length() const
{
    return m_writer -> builder() -> length();
}

void ArrowBackedArrayBuilder::handleRowFinished()
{
    if( m_hasExternalValue )
    {
        m_writer -> writeNext( m_externalStruct );
        m_hasExternalValue = false;
        m_externalStruct = nullptr;
    }
    else
    {
        m_writer -> writeNull();
    }
}

std::shared_ptr<::arrow::Array> ArrowBackedArrayBuilder::buildArray()
{
    auto arrays = m_writer -> finish();
    CSP_TRUE_OR_THROW_RUNTIME( arrays.size() == 1,
        "ArrowBackedArrayBuilder expected 1 array from FieldWriter, got " << arrays.size() );
    // finish() resets the underlying builder's capacity; re-reserve for the next chunk so the per-row
    // append path stays allocation-free (and keeps UnsafeAppend within reserved capacity).
    m_writer -> reserve( getChunkSize() );
    return arrays[0];
}

std::shared_ptr<ArrowBackedArrayBuilder> createArrowBackedArrayBuilderForField(
    const std::string & columnName, std::uint32_t chunkSize,
    const StructFieldPtr & structField )
{
    return std::make_shared<ArrowBackedArrayBuilder>( columnName, chunkSize, structField );
}

} // namespace csp::adapters::parquet
