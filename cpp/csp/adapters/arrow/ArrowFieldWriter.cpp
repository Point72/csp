// Concrete FieldWriter implementations for all CSP scalar types.
//
// Fixed-length writers use UnsafeWriter<BuilderT> — a single template
// that takes a value-extraction callable at construction (mirrors LambdaReader).
// Variable-length writers (StringLike, Enum) and NestedStruct are separate
// classes because they need safe Append (variable-length) or recursive logic.

#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/engine/CspType.h>
#include <csp/engine/CspEnum.h>

#include <arrow/builder.h>
#include <arrow/type.h>

namespace csp::adapters::arrow
{

#define ARROW_OK_OR_THROW( expr, msg ) \
    do { auto __s = ( expr ); if( !__s.ok() ) CSP_THROW( RuntimeException, msg << ": " << __s.ToString() ); } while(0)

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

// --- Generic lambda-based writer for fixed-length types ---
// ValueFn signature: auto(const Struct *) — returns the value to UnsafeAppend/Append.
// Covers: all numeric primitives, bool, DateTime, TimeDelta, Time, Date.

template<typename ArrowBuilderT, typename ValueFn>
class UnsafeWriter final : public FieldWriter
{
public:
    UnsafeWriter( const std::string & columnName, const StructFieldPtr & field,
                  std::shared_ptr<ArrowBuilderT> typedBuilder,
                  std::shared_ptr<::arrow::DataType> dataType, ValueFn fn )
        : FieldWriter( columnName, field, typedBuilder, std::move( dataType ) ),
          m_typedBuilder( typedBuilder.get() ), m_fn( std::move( fn ) ) {}

    void writeAll( const std::vector<StructPtr> & structs, int64_t offset, int64_t count ) override
    {
        for( int64_t i = offset; i < offset + count; ++i )
        {
            const Struct * s = structs[i].get();
            if( m_field -> isSet( s ) )
                m_typedBuilder -> UnsafeAppend( m_fn( s ) );
            else
                m_typedBuilder -> UnsafeAppendNull();
        }
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( m_fn( s ) ), "Failed to append value" );
    }

private:
    ArrowBuilderT * m_typedBuilder;
    ValueFn         m_fn;
};

// Factory: creates an UnsafeWriter, deducing ValueFn type, and returns CreatedFieldWriter
template<typename ArrowBuilderT, typename ValueFn>
CreatedFieldWriter makeUnsafeWriter( const std::string & name, const StructFieldPtr & field,
                                     std::shared_ptr<ArrowBuilderT> builder,
                                     std::shared_ptr<::arrow::DataType> dataType, ValueFn && fn )
{
    auto w = std::make_unique<UnsafeWriter<ArrowBuilderT, std::decay_t<ValueFn>>>(
        name, field, builder, std::move( dataType ), std::forward<ValueFn>( fn ) );
    return { std::move( w ), std::move( builder ) };
}

// Factory: primitive numeric writer (auto-creates builder from default constructor)
template<typename CspT, typename ArrowBuilderT>
CreatedFieldWriter makePrimitiveWriter( const std::string & name, const StructFieldPtr & f )
{
    auto b = std::make_shared<ArrowBuilderT>();
    return makeUnsafeWriter( name, f, b, b -> type(), [f]( const Struct * s ) {
        return static_cast<typename ArrowBuilderT::value_type>( f -> value<CspT>( s ) );
    } );
}

// Factory: nanosecond-based temporal writer (DateTime, TimeDelta, Time)
template<typename CspT, typename ArrowBuilderT>
CreatedFieldWriter makeNanosWriter( const std::string & name, const StructFieldPtr & f,
                                    std::shared_ptr<::arrow::DataType> dataType )
{
    auto b = std::make_shared<ArrowBuilderT>( dataType, ::arrow::default_memory_pool() );
    return makeUnsafeWriter( name, f, b, std::move( dataType ), [f]( const Struct * s ) {
        return f -> value<CspT>( s ).asNanoseconds();
    } );
}

// --- String / Bytes writer (variable-length: needs safe Append) ---

template<typename ArrowBuilderT>
class StringLikeWriter final : public FieldWriter
{
public:
    StringLikeWriter( const std::string & columnName, const StructFieldPtr & field,
                      std::shared_ptr<::arrow::DataType> dataType )
        : FieldWriter( columnName, field, std::make_shared<ArrowBuilderT>(), std::move( dataType ) ),
          m_typedBuilder( static_cast<ArrowBuilderT *>( m_builder.get() ) ) {}

    void writeAll( const std::vector<StructPtr> & structs, int64_t offset, int64_t count ) override
    {
        for( int64_t i = offset; i < offset + count; ++i )
        {
            const Struct * s = structs[i].get();
            if( m_field -> isSet( s ) )
            {
                auto & val = m_field -> value<std::string>( s );
                ARROW_OK_OR_THROW( m_typedBuilder -> Append( val.c_str(), val.length() ), "Failed to append string/bytes" );
            }
            else
                ARROW_OK_OR_THROW( m_typedBuilder -> AppendNull(), "Failed to append null" );
        }
    }

protected:
    void doWrite( const Struct * s ) override
    {
        auto & val = m_field -> value<std::string>( s );
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( val.c_str(), val.length() ), "Failed to append string/bytes" );
    }

private:
    ArrowBuilderT * m_typedBuilder;
};

// --- Enum writer (variable-length string: CspEnum → name()) ---

class EnumWriter final : public FieldWriter
{
public:
    EnumWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field, std::make_shared<::arrow::StringBuilder>(), ::arrow::utf8() ),
          m_typedBuilder( static_cast<::arrow::StringBuilder *>( m_builder.get() ) ) {}

    void writeAll( const std::vector<StructPtr> & structs, int64_t offset, int64_t count ) override
    {
        for( int64_t i = offset; i < offset + count; ++i )
        {
            const Struct * s = structs[i].get();
            if( m_field -> isSet( s ) )
            {
                auto & n = m_field -> value<CspEnum>( s ).name();
                ARROW_OK_OR_THROW( m_typedBuilder -> Append( n.c_str(), n.length() ), "Failed to append enum" );
            }
            else
                ARROW_OK_OR_THROW( m_typedBuilder -> AppendNull(), "Failed to append null" );
        }
    }

protected:
    void doWrite( const Struct * s ) override
    {
        auto & n = m_field -> value<CspEnum>( s ).name();
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( n.c_str(), n.length() ), "Failed to append enum" );
    }

private:
    ::arrow::StringBuilder * m_typedBuilder;
};

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
        ARROW_OK_OR_THROW( m_structBuilder -> AppendNull(), "Failed to append null struct" );
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

template<typename WriterT, typename... Args>
CreatedFieldWriter makeWriter( Args &&... args )
{
    auto w = std::make_unique<WriterT>( std::forward<Args>( args )... );
    auto b = w -> builder();
    return { std::move( w ), std::move( b ) };
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
        case CspType::Type::BOOL:
        {
            auto b = std::make_shared<::arrow::BooleanBuilder>();
            return makeUnsafeWriter( columnName, f, b, ::arrow::boolean(),
                [f]( const Struct * s ) { return f -> value<bool>( s ); } );
        }
        case CspType::Type::INT8:   return makePrimitiveWriter<int8_t,   ::arrow::Int8Builder>( columnName, f );
        case CspType::Type::INT16:  return makePrimitiveWriter<int16_t,  ::arrow::Int16Builder>( columnName, f );
        case CspType::Type::INT32:  return makePrimitiveWriter<int32_t,  ::arrow::Int32Builder>( columnName, f );
        case CspType::Type::INT64:  return makePrimitiveWriter<int64_t,  ::arrow::Int64Builder>( columnName, f );
        case CspType::Type::UINT8:  return makePrimitiveWriter<uint8_t,  ::arrow::UInt8Builder>( columnName, f );
        case CspType::Type::UINT16: return makePrimitiveWriter<uint16_t, ::arrow::UInt16Builder>( columnName, f );
        case CspType::Type::UINT32: return makePrimitiveWriter<uint32_t, ::arrow::UInt32Builder>( columnName, f );
        case CspType::Type::UINT64: return makePrimitiveWriter<uint64_t, ::arrow::UInt64Builder>( columnName, f );
        case CspType::Type::DOUBLE: return makePrimitiveWriter<double,   ::arrow::DoubleBuilder>( columnName, f );

        // --- String / Bytes ---
        case CspType::Type::STRING:
            if( isBytesField( f ) )
                return makeWriter<StringLikeWriter<::arrow::BinaryBuilder>>( columnName, f, ::arrow::binary() );
            return makeWriter<StringLikeWriter<::arrow::StringBuilder>>( columnName, f, ::arrow::utf8() );

        case CspType::Type::ENUM: return makeWriter<EnumWriter>( columnName, f );

        // --- Temporal ---
        case CspType::Type::DATETIME:
            return makeNanosWriter<DateTime, ::arrow::TimestampBuilder>(
                columnName, f, std::make_shared<::arrow::TimestampType>( ::arrow::TimeUnit::NANO, "UTC" ) );
        case CspType::Type::TIMEDELTA:
            return makeNanosWriter<TimeDelta, ::arrow::DurationBuilder>(
                columnName, f, std::make_shared<::arrow::DurationType>( ::arrow::TimeUnit::NANO ) );
        case CspType::Type::TIME:
            return makeNanosWriter<Time, ::arrow::Time64Builder>(
                columnName, f, std::make_shared<::arrow::Time64Type>( ::arrow::TimeUnit::NANO ) );

        // --- Date (days since epoch) ---
        case CspType::Type::DATE:
        {
            auto b = std::make_shared<::arrow::Date32Builder>();
            return makeUnsafeWriter( columnName, f, b, ::arrow::date32(), [f]( const Struct * s ) {
                auto & d = f -> value<Date>( s );
                return static_cast<int32_t>( DateTime( d.year(), d.month(), d.day() ).asNanoseconds() / csp::NANOS_PER_DAY );
            } );
        }

        // --- Nested struct ---
        case CspType::Type::STRUCT:
        {
            auto nestedMeta = std::static_pointer_cast<const CspStructType>( f -> type() ) -> meta();

            std::vector<std::shared_ptr<::arrow::Field>> arrowFields;
            std::vector<std::shared_ptr<::arrow::ArrayBuilder>> childBuilders;
            std::vector<std::unique_ptr<FieldWriter>> childWriters;

            // Use fieldNames() for stable insertion order (fields() is sorted for memory layout)
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

}
