#include <csp/adapters/parquet/ArrowBackedArrayBuilder.h>
#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/adapters/arrow/ArrowTypeVisitor.h>

namespace csp::adapters::parquet
{

static StructFieldPtr createStructFieldFromCspType( const std::string & name, CspTypePtr cspType )
{
    if( cspType -> type() == CspType::Type::STRING )
        return std::make_shared<StringStructField>( CspType::STRING(), name, false );
    if( cspType -> type() == CspType::Type::ENUM )
        return std::make_shared<CspEnumStructField>( cspType, name, false );

    return csp::adapters::arrow::visitCspValueType( cspType -> type(),
        [&]( auto tag ) -> StructFieldPtr
        {
            using T = typename decltype( tag )::type;
            if constexpr( std::is_same_v<T, std::string> || std::is_same_v<T, CspEnum> )
                return {};  // unreachable; STRING/ENUM handled above
            else
                return std::make_shared<NativeStructField<T>>( name, false );
        },
        [&]() -> StructFieldPtr
        {
            CSP_THROW( TypeError, "Unsupported CSP type for struct field: " << cspType -> type() );
        } );
}

ScratchFieldInfo createScratchField( const std::string & name, CspTypePtr cspType )
{
    auto field = createStructFieldFromCspType( name, cspType );

    auto meta = std::make_shared<StructMeta>(
        "__scratch_" + name,
        StructMeta::Fields{ field },
        false
    );

    return ScratchFieldInfo{ meta, meta -> field( name ) };
}

ArrowBackedArrayBuilder::ArrowBackedArrayBuilder(
    const std::string & columnName, std::uint32_t chunkSize,
    CspTypePtr cspType, bool isBytes )
    : ArrowSingleColumnArrayBuilder( columnName, chunkSize )
    , m_isScratchMode( true )
{
    // For bytes fields, we need to create a STRING type scratch field
    // but the FieldWriter should produce binary output.
    // The createScratchField always creates STRING type fields for STRING cspType.
    auto scratch = createScratchField( columnName, cspType );
    m_scratchMeta = scratch.meta;
    m_field       = scratch.field;
    m_scratch     = m_scratchMeta -> create();

    // If isBytes, wrap the STRING field so FieldWriter produces binary
    if( isBytes )
    {
        auto bytesField = std::make_shared<StringStructField>(
            std::make_shared<CspStringType>( true ), columnName, false );
        auto bytesMeta = std::make_shared<StructMeta>(
            "__scratch_bytes_" + columnName,
            StructMeta::Fields{ bytesField },
            false
        );
        m_scratchMeta = bytesMeta;
        m_field       = bytesMeta -> field( columnName );
        m_scratch     = m_scratchMeta -> create();
    }

    init( columnName, m_field );
}

ArrowBackedArrayBuilder::ArrowBackedArrayBuilder(
    const std::string & columnName, std::uint32_t chunkSize,
    const StructFieldPtr & structField )
    : ArrowSingleColumnArrayBuilder( columnName, chunkSize )
    , m_field( structField )
    , m_isScratchMode( false )
{
    init( columnName, structField );
}

ArrowBackedArrayBuilder::~ArrowBackedArrayBuilder() = default;

void ArrowBackedArrayBuilder::init( const std::string & columnName, const StructFieldPtr & field )
{
    auto created = csp::adapters::arrow::createFieldWriter( columnName, field );
    m_writer = std::move( created.writer );
}

std::shared_ptr<::arrow::DataType> ArrowBackedArrayBuilder::getDataType()
{
    return m_writer -> dataTypes()[0];
}

std::shared_ptr<::arrow::ArrayBuilder> ArrowBackedArrayBuilder::getBuilder()
{
    return m_writer -> builder();
}

int64_t ArrowBackedArrayBuilder::length() const
{
    return m_writer -> builder() -> length();
}

void ArrowBackedArrayBuilder::handleRowFinished()
{
    if( m_isScratchMode )
    {
        // Scratch mode: FieldWriter reads from scratch struct.
        // writeNext checks isSet internally and appends value or null.
        m_writer -> writeNext( m_scratch.get() );

        // Clear the isSet bit so next cycle starts with null
        uint8_t * mask = reinterpret_cast<uint8_t *>( m_scratch.get() ) + m_field -> maskOffset();
        *mask &= ~m_field -> maskBitMask();
    }
    else
    {
        // External mode: read from externally-provided struct
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
}

std::shared_ptr<::arrow::Array> ArrowBackedArrayBuilder::buildArray()
{
    auto arrays = m_writer -> finish();
    CSP_TRUE_OR_THROW_RUNTIME( arrays.size() == 1,
        "ArrowBackedArrayBuilder expected 1 array from FieldWriter, got " << arrays.size() );
    return arrays[0];
}

std::shared_ptr<ArrowBackedArrayBuilder> createArrowBackedArrayBuilder(
    const std::string & columnName, std::uint32_t chunkSize,
    CspTypePtr cspType, bool isBytes )
{
    return std::make_shared<ArrowBackedArrayBuilder>( columnName, chunkSize, cspType, isBytes );
}

std::shared_ptr<ArrowBackedArrayBuilder> createArrowBackedArrayBuilderForField(
    const std::string & columnName, std::uint32_t chunkSize,
    const StructFieldPtr & structField )
{
    return std::make_shared<ArrowBackedArrayBuilder>( columnName, chunkSize, structField );
}

} // namespace csp::adapters::parquet
