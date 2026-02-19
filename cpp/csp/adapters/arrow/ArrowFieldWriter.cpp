// Concrete FieldWriter implementations for all CSP scalar types.
//
// Each writer extends FieldWriter directly (no intermediate base).
// The base provides default reserve/writeNext/writeNull/finish.
// Concrete writers only implement doWrite().

#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/engine/CspType.h>
#include <csp/engine/CspEnum.h>

#include <arrow/builder.h>
#include <arrow/type.h>

namespace csp::adapters::arrow
{

// Helper macro to check arrow Status
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

// --- Primitive numeric writers (int/uint/double/bool) ---

template<typename CspT, typename ArrowBuilderT>
class PrimitiveWriter final : public FieldWriter
{
public:
    PrimitiveWriter( const std::string & columnName, const StructFieldPtr & field,
                     std::shared_ptr<ArrowBuilderT> typedBuilder )
        : FieldWriter( columnName, field, typedBuilder, typedBuilder -> type() ),
          m_typedBuilder( typedBuilder.get() )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW(
            m_typedBuilder -> Append( static_cast<typename ArrowBuilderT::value_type>( m_field -> value<CspT>( s ) ) ),
            "Failed to append primitive value" );
    }

private:
    ArrowBuilderT * m_typedBuilder;
};

// Bool specialization (BooleanBuilder::value_type != bool)
class BoolWriter final : public FieldWriter
{
public:
    BoolWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::BooleanBuilder>(),
                       ::arrow::boolean() ),
          m_typedBuilder( static_cast<::arrow::BooleanBuilder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( m_field -> value<bool>( s ) ), "Failed to append bool" );
    }

private:
    ::arrow::BooleanBuilder * m_typedBuilder;
};

// --- String writer ---

class StringWriter final : public FieldWriter
{
public:
    StringWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::StringBuilder>(),
                       ::arrow::utf8() ),
          m_typedBuilder( static_cast<::arrow::StringBuilder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        auto & val = m_field -> value<std::string>( s );
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( val.c_str(), val.length() ), "Failed to append string" );
    }

private:
    ::arrow::StringBuilder * m_typedBuilder;
};

// --- Bytes writer ---

class BytesWriter final : public FieldWriter
{
public:
    BytesWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::BinaryBuilder>(),
                       ::arrow::binary() ),
          m_typedBuilder( static_cast<::arrow::BinaryBuilder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        auto & val = m_field -> value<std::string>( s );
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( val.c_str(), val.length() ), "Failed to append bytes" );
    }

private:
    ::arrow::BinaryBuilder * m_typedBuilder;
};

// --- Enum writer (writes as string) ---

class EnumWriter final : public FieldWriter
{
public:
    EnumWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::StringBuilder>(),
                       ::arrow::utf8() ),
          m_typedBuilder( static_cast<::arrow::StringBuilder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        auto & name = m_field -> value<CspEnum>( s ).name();
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( name.c_str(), name.length() ), "Failed to append enum" );
    }

private:
    ::arrow::StringBuilder * m_typedBuilder;
};

// --- DateTime writer (TIMESTAMP ns UTC) ---

class DateTimeWriter final : public FieldWriter
{
public:
    DateTimeWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::TimestampBuilder>(
                           std::make_shared<::arrow::TimestampType>( ::arrow::TimeUnit::NANO, "UTC" ),
                           ::arrow::default_memory_pool() ),
                       std::make_shared<::arrow::TimestampType>( ::arrow::TimeUnit::NANO, "UTC" ) ),
          m_typedBuilder( static_cast<::arrow::TimestampBuilder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( m_field -> value<DateTime>( s ).asNanoseconds() ),
                           "Failed to append datetime" );
    }

private:
    ::arrow::TimestampBuilder * m_typedBuilder;
};

// --- TimeDelta writer (DURATION ns) ---

class TimeDeltaWriter final : public FieldWriter
{
public:
    TimeDeltaWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::DurationBuilder>(
                           std::make_shared<::arrow::DurationType>( ::arrow::TimeUnit::NANO ),
                           ::arrow::default_memory_pool() ),
                       std::make_shared<::arrow::DurationType>( ::arrow::TimeUnit::NANO ) ),
          m_typedBuilder( static_cast<::arrow::DurationBuilder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( m_field -> value<TimeDelta>( s ).asNanoseconds() ),
                           "Failed to append timedelta" );
    }

private:
    ::arrow::DurationBuilder * m_typedBuilder;
};

// --- Date writer (DATE32 days since epoch) ---

class DateWriter final : public FieldWriter
{
public:
    DateWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::Date32Builder>(),
                       ::arrow::date32() ),
          m_typedBuilder( static_cast<::arrow::Date32Builder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        // Compute days since Unix epoch directly via a single DateTime construction.
        // Date::operator- would call timegm() twice (once for the date, once for epoch);
        // constructing DateTime directly calls timegm() only once.
        auto & date = m_field -> value<Date>( s );
        DateTime dt( date.year(), date.month(), date.day() );
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( static_cast<int32_t>( dt.asNanoseconds() / csp::NANOS_PER_DAY ) ),
                           "Failed to append date" );
    }

private:
    ::arrow::Date32Builder * m_typedBuilder;
};

// --- Time writer (TIME64 ns) ---

class TimeWriter final : public FieldWriter
{
public:
    TimeWriter( const std::string & columnName, const StructFieldPtr & field )
        : FieldWriter( columnName, field,
                       std::make_shared<::arrow::Time64Builder>(
                           std::make_shared<::arrow::Time64Type>( ::arrow::TimeUnit::NANO ),
                           ::arrow::default_memory_pool() ),
                       std::make_shared<::arrow::Time64Type>( ::arrow::TimeUnit::NANO ) ),
          m_typedBuilder( static_cast<::arrow::Time64Builder *>( m_builder.get() ) )
    {
    }

protected:
    void doWrite( const Struct * s ) override
    {
        ARROW_OK_OR_THROW( m_typedBuilder -> Append( m_field -> value<Time>( s ).asNanoseconds() ),
                           "Failed to append time" );
    }

private:
    ::arrow::Time64Builder * m_typedBuilder;
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
          m_childWriters( std::move( childWriters ) )
    {
    }

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

protected:
    void doWrite( const Struct * s ) override
    {
        auto & nested = m_field -> value<StructPtr>( s );
        for( auto & cw : m_childWriters )
            cw -> writeNext( nested.get() );
        ARROW_OK_OR_THROW( m_structBuilder -> Append(), "Failed to append struct" );
    }

private:
    ::arrow::StructBuilder *                      m_structBuilder;
    std::vector<std::unique_ptr<FieldWriter>>     m_childWriters;
};

// Helper: detect bytes vs string for CspStringType
bool isBytesField( const StructFieldPtr & field )
{
    if( field -> type() -> type() != CspType::Type::STRING )
        return false;
    auto strType = std::static_pointer_cast<const CspStringType>( field -> type() );
    return strType && strType -> isBytes();
}

// Helper: create a PrimitiveWriter and return CreatedFieldWriter
template<typename CspT, typename ArrowBuilderT>
CreatedFieldWriter makePrimitive( const std::string & name, const StructFieldPtr & field )
{
    auto b = std::make_shared<ArrowBuilderT>();
    auto w = std::make_unique<PrimitiveWriter<CspT, ArrowBuilderT>>( name, field, b );
    return { std::move( w ), std::move( b ) };
}

// Helper: create any writer and return CreatedFieldWriter
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
    auto cspType = structField -> type() -> type();

    switch( cspType )
    {
        case CspType::Type::BOOL:
            return makeWriter<BoolWriter>( columnName, structField );

        case CspType::Type::INT8:
            return makePrimitive<int8_t, ::arrow::Int8Builder>( columnName, structField );
        case CspType::Type::INT16:
            return makePrimitive<int16_t, ::arrow::Int16Builder>( columnName, structField );
        case CspType::Type::INT32:
            return makePrimitive<int32_t, ::arrow::Int32Builder>( columnName, structField );
        case CspType::Type::INT64:
            return makePrimitive<int64_t, ::arrow::Int64Builder>( columnName, structField );

        case CspType::Type::UINT8:
            return makePrimitive<uint8_t, ::arrow::UInt8Builder>( columnName, structField );
        case CspType::Type::UINT16:
            return makePrimitive<uint16_t, ::arrow::UInt16Builder>( columnName, structField );
        case CspType::Type::UINT32:
            return makePrimitive<uint32_t, ::arrow::UInt32Builder>( columnName, structField );
        case CspType::Type::UINT64:
            return makePrimitive<uint64_t, ::arrow::UInt64Builder>( columnName, structField );

        case CspType::Type::DOUBLE:
            return makePrimitive<double, ::arrow::DoubleBuilder>( columnName, structField );

        case CspType::Type::STRING:
        {
            if( isBytesField( structField ) )
                return makeWriter<BytesWriter>( columnName, structField );
            return makeWriter<StringWriter>( columnName, structField );
        }

        case CspType::Type::ENUM:
            return makeWriter<EnumWriter>( columnName, structField );

        case CspType::Type::DATETIME:
            return makeWriter<DateTimeWriter>( columnName, structField );

        case CspType::Type::TIMEDELTA:
            return makeWriter<TimeDeltaWriter>( columnName, structField );

        case CspType::Type::DATE:
            return makeWriter<DateWriter>( columnName, structField );

        case CspType::Type::TIME:
            return makeWriter<TimeWriter>( columnName, structField );

        case CspType::Type::STRUCT:
        {
            auto nestedMeta = std::static_pointer_cast<const CspStructType>( structField -> type() ) -> meta();

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
            auto dataType = std::static_pointer_cast<::arrow::DataType>( structType );

            auto w = std::make_unique<NestedStructWriter>(
                columnName, structField, structBuilder, std::move( dataType ), std::move( childWriters ) );
            return { std::move( w ), std::move( structBuilder ) };
        }

        default:
            CSP_THROW( TypeError, "Unsupported CSP type " << cspType << " for field '" << columnName << "'" );
    }
}

#undef ARROW_OK_OR_THROW

}
