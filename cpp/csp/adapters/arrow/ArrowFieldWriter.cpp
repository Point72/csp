// Concrete FieldWriter implementations for all CSP scalar types.
//
// Every scalar/temporal/string/enum column is a ScalarFieldWriter<CspT>: it reads the value out of
// a struct field and appends it through the shared ArrowScalarColumnWriter<CspT> kernel (the single
// home of the per-type Arrow-builder choice + conversion, also used value-first by the scalar output
// path). NestedStruct is a separate class because it is recursive.

#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/adapters/arrow/ArrowScalarWriter.h>
#include <csp/engine/CspType.h>
#include <csp/engine/CspEnum.h>

#include <arrow/builder.h>
#include <arrow/type.h>

namespace csp::adapters::arrow
{

#define ARROW_OK_OR_THROW( expr, msg ) \
    do { auto _st = ( expr ); if( !_st.ok() ) CSP_THROW( RuntimeException, msg << ": " << _st.ToString() ); } while(0)

// --- Base class default implementations ---

void FieldWriter::reserve( int64_t numRows )
{
    ARROW_OK_OR_THROW( m_builder -> Reserve( numRows ), "Failed to reserve builder capacity" );
}

void FieldWriter::writeNext( const Struct * s )
{
    if( m_field -> isSet( s ) )
        doWrite( s );
    else
        writeNull();
}

void FieldWriter::writeAll( const std::vector<StructPtr> & structs, int64_t offset, int64_t count )
{
    for( int64_t i = offset; i < offset + count; ++i )
        writeNext( structs[i].get() );
}

void FieldWriter::writeNull()
{
    ARROW_OK_OR_THROW( m_builder -> AppendNull(), "Failed to append null" );
}

std::vector<std::shared_ptr<::arrow::Array>> FieldWriter::finish()
{
    std::shared_ptr<::arrow::Array> arr;
    ARROW_OK_OR_THROW( m_builder -> Finish( &arr ), "Failed to finish array" );
    return { arr };
}

namespace
{

// --- Scalar field writer: reads field->value<CspT>(s), appends via the shared kernel ---
// Replaces the former per-category writers (numeric/temporal/string/enum). The per-type Arrow
// builder + value conversion now live exactly once, in ArrowScalarColumnWriter<CspT>.

template<typename CspT>
class ScalarFieldWriter final : public FieldWriter
{
public:
    ScalarFieldWriter( const std::string & columnName, const StructFieldPtr & field, bool isBytes = false )
        : ScalarFieldWriter( columnName, field, ArrowScalarColumnWriter<CspT>( isBytes ) )
    {
    }

    // Columnar bulk path: caller has reserved, so set values use the unsafe append.
    void writeAll( const std::vector<StructPtr> & structs, int64_t offset, int64_t count ) override
    {
        for( int64_t i = offset; i < offset + count; ++i )
        {
            const Struct * s = structs[i].get();
            if( m_field -> isSet( s ) )
                m_kernel.appendUnsafe( m_field -> value<CspT>( s ) );
            else
                m_kernel.appendNull();
        }
    }

protected:
    void doWrite( const Struct * s ) override
    {
        m_kernel.append( m_field -> value<CspT>( s ) );
    }

private:
    // Delegating ctor: build the kernel first so its builder/dataType can seed the FieldWriter base,
    // then adopt the kernel (both hold the same shared builder).
    ScalarFieldWriter( const std::string & columnName, const StructFieldPtr & field,
                       ArrowScalarColumnWriter<CspT> kernel )
        : FieldWriter( columnName, field, kernel.builder(), kernel.dataType() ),
          m_kernel( std::move( kernel ) )
    {
    }

    ArrowScalarColumnWriter<CspT> m_kernel;
};

template<typename CspT>
CreatedFieldWriter makeScalarFieldWriter( const std::string & name, const StructFieldPtr & f, bool isBytes = false )
{
    auto w = std::make_unique<ScalarFieldWriter<CspT>>( name, f, isBytes );
    auto b = w -> builder();
    return { std::move( w ), std::move( b ) };
}

// --- Nested struct writer (recursive) ---

class NestedStructWriter final : public FieldWriter
{
public:
    NestedStructWriter( const std::string & columnName, const StructFieldPtr & field,
                        std::shared_ptr<::arrow::StructBuilder> structBuilder,
                        std::shared_ptr<::arrow::DataType> structType,
                        std::vector<std::unique_ptr<FieldWriter>> childWriters )
        : FieldWriter( columnName, field, structBuilder, std::move( structType ) ),
          m_structBuilder( structBuilder.get() ),
          m_childWriters( std::move( childWriters ) ) {}

    void reserve( int64_t numRows ) override
    {
        ARROW_OK_OR_THROW( m_builder -> Reserve( numRows ), "Failed to reserve builder capacity" );
        for( auto & cw : m_childWriters )
            cw -> reserve( numRows );
    }

    void writeNull() override
    {
        for( auto & cw : m_childWriters )
            cw -> writeNull();
        // Append(false) sets the struct validity bit without touching children
        // (unlike AppendNull which also calls AppendEmptyValue on each child).
        ARROW_OK_OR_THROW( m_structBuilder -> Append( false ), "Failed to append null struct" );
    }

    void writeAll( const std::vector<StructPtr> & structs, int64_t offset, int64_t count ) override
    {
        // Check if any parent struct has a null nested value
        bool hasNulls = false;
        for( int64_t i = offset; i < offset + count && !hasNulls; ++i )
            hasNulls = !m_field -> isSet( structs[i].get() );

        if( !hasNulls )
        {
            // Fast path: all nested values are set — columnar child writes
            std::vector<StructPtr> nested( count );
            for( int64_t i = 0; i < count; ++i )
                nested[i] = m_field -> value<StructPtr>( structs[offset + i].get() );
            for( auto & cw : m_childWriters )
                cw -> writeAll( nested, 0, count );
            ARROW_OK_OR_THROW( m_structBuilder -> AppendValues( count, nullptr ), "Failed to append struct validity" );
        }
        else
        {
            for( int64_t i = offset; i < offset + count; ++i )
                writeNext( structs[i].get() );
        }
    }

protected:
    void doWrite( const Struct * s ) override
    {
        auto & nested = m_field -> value<StructPtr>( s );
        for( auto & cw : m_childWriters )
            cw -> writeNext( nested.get() );
        ARROW_OK_OR_THROW( m_structBuilder -> Append(), "Failed to append struct" );
    }

private:
    ::arrow::StructBuilder *                  m_structBuilder;
    std::vector<std::unique_ptr<FieldWriter>> m_childWriters;
};

// --- Factory helpers ---

bool isBytesField( const StructFieldPtr & field )
{
    if( field -> type() -> type() != CspType::Type::STRING )
        return false;
    auto strType = std::static_pointer_cast<const CspStringType>( field -> type() );
    return strType && strType -> isBytes();
}

} // anonymous namespace

CreatedFieldWriter createFieldWriter(
    const std::string & columnName,
    const StructFieldPtr & structField )
{
    auto & f = structField;

    switch( f -> type() -> type() )
    {
        // --- Numeric ---
        case CspType::Type::BOOL:   return makeScalarFieldWriter<bool>(     columnName, f );
        case CspType::Type::INT8:   return makeScalarFieldWriter<int8_t>(   columnName, f );
        case CspType::Type::INT16:  return makeScalarFieldWriter<int16_t>(  columnName, f );
        case CspType::Type::INT32:  return makeScalarFieldWriter<int32_t>(  columnName, f );
        case CspType::Type::INT64:  return makeScalarFieldWriter<int64_t>(  columnName, f );
        case CspType::Type::UINT8:  return makeScalarFieldWriter<uint8_t>(  columnName, f );
        case CspType::Type::UINT16: return makeScalarFieldWriter<uint16_t>( columnName, f );
        case CspType::Type::UINT32: return makeScalarFieldWriter<uint32_t>( columnName, f );
        case CspType::Type::UINT64: return makeScalarFieldWriter<uint64_t>( columnName, f );
        case CspType::Type::DOUBLE: return makeScalarFieldWriter<double>(   columnName, f );

        // --- String / Bytes ---
        case CspType::Type::STRING: return makeScalarFieldWriter<std::string>( columnName, f, isBytesField( f ) );

        case CspType::Type::ENUM:   return makeScalarFieldWriter<CspEnum>( columnName, f );

        // --- Temporal ---
        case CspType::Type::DATETIME:  return makeScalarFieldWriter<DateTime>(  columnName, f );
        case CspType::Type::TIMEDELTA: return makeScalarFieldWriter<TimeDelta>( columnName, f );
        case CspType::Type::TIME:      return makeScalarFieldWriter<Time>(      columnName, f );

        // --- Date (days since epoch) ---
        case CspType::Type::DATE:      return makeScalarFieldWriter<Date>(      columnName, f );

        // --- Nested struct ---
        case CspType::Type::STRUCT:
        {
            auto nestedMeta = std::static_pointer_cast<const CspStructType>( f -> type() ) -> meta();

            std::vector<std::shared_ptr<::arrow::Field>> arrowFields;
            std::vector<std::shared_ptr<::arrow::ArrayBuilder>> childBuilders;
            std::vector<std::unique_ptr<FieldWriter>> childWriters;

            // Ordering convention: csp writes struct child columns in declaration order, using
            // fieldNames() rather than fields() (which is sorted by struct memory layout). This keeps
            // nested-struct child order consistent with top-level column order and stable against
            // internal field packing. The reader matches children by name, so either order round-trips,
            // but declaration order is the on-disk contract.
            for( auto & subFieldName : nestedMeta -> fieldNames() )
            {
                auto subField = nestedMeta -> field( subFieldName );
                auto child = createFieldWriter( subFieldName, subField );
                arrowFields.push_back( std::make_shared<::arrow::Field>( subFieldName, child.writer -> dataTypes()[0] ) );
                childBuilders.push_back( std::move( child.builder ) );
                childWriters.push_back( std::move( child.writer ) );
            }

            auto structType = std::make_shared<::arrow::StructType>( arrowFields );
            auto structBuilder = std::make_shared<::arrow::StructBuilder>(
                structType, ::arrow::default_memory_pool(), childBuilders );

            auto w = std::make_unique<NestedStructWriter>(
                columnName, f, structBuilder,
                std::static_pointer_cast<::arrow::DataType>( structType ), std::move( childWriters ) );
            return { std::move( w ), std::move( structBuilder ) };
        }

        default:
            CSP_THROW( TypeError, "Unsupported CSP type " << f -> type() -> type()
                                   << " for field '" << columnName << "'" );
    }
}

#undef ARROW_OK_OR_THROW

// --- List field writer factory ---

static ListFieldWriterFactory s_listFieldWriterFactory;

void registerListFieldWriterFactory( ListFieldWriterFactory factory )
{
    s_listFieldWriterFactory = std::move( factory );
}

std::pair<std::shared_ptr<::arrow::ArrayBuilder>, ListItemsWriter> createListFieldWriter( const CspTypePtr & elemType )
{
    CSP_TRUE_OR_THROW_RUNTIME( s_listFieldWriterFactory,
        "No list field writer factory registered (numpy support not loaded?)" );
    return s_listFieldWriterFactory( elemType );
}

}
