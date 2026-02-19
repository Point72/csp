// Concrete FieldReader implementations for all scalar Arrow types.
//
// Each reader extends FieldReader directly and implements doReadNext().
// NestedStructReader binds children to child arrays and uses readNext()/skipNext().

#include <csp/adapters/arrow/ArrowFieldReader.h>
#include <csp/engine/CspType.h>
#include <csp/engine/CspEnum.h>

#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/util/float16.h>

namespace csp::adapters::arrow
{

namespace
{

// Helper: compute nanosecond multiplier for a given arrow::TimeUnit
int64_t timeUnitMultiplier( ::arrow::TimeUnit::type unit )
{
    switch( unit )
    {
        case ::arrow::TimeUnit::SECOND: return csp::NANOS_PER_SECOND;
        case ::arrow::TimeUnit::MILLI:  return csp::NANOS_PER_MILLISECOND;
        case ::arrow::TimeUnit::MICRO:  return csp::NANOS_PER_MICROSECOND;
        case ::arrow::TimeUnit::NANO:   return 1LL;
    }
    CSP_THROW( TypeError, "Unexpected arrow TimeUnit: " << static_cast<int>( unit ) );
}

// --- Primitive numeric readers (int/uint/float/bool) ---

template<typename CspT, typename ArrowArrayT>
class PrimitiveReader final : public FieldReader
{
public:
    using FieldReader::FieldReader;

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ArrowArrayT &>( *m_column );
        if( typed.IsValid( row ) )
            m_field -> setValue<CspT>( s, static_cast<CspT>( typed.Value( row ) ) );
    }
};

// --- Half-float reader (special: Value() returns uint16 bits, needs Float16 conversion) ---

class HalfFloatReader final : public FieldReader
{
public:
    using FieldReader::FieldReader;

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::HalfFloatArray &>( *m_column );
        if( typed.IsValid( row ) )
            m_field -> setValue<double>( s, ::arrow::util::Float16::FromBits( typed.Value( row ) ).ToDouble() );
    }
};

// --- String readers ---

template<typename ArrowStringArrayT>
class StringReader final : public FieldReader
{
public:
    using FieldReader::FieldReader;

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ArrowStringArrayT &>( *m_column );
        if( typed.IsValid( row ) )
        {
            auto view = typed.GetView( row );
            m_field -> setValue<std::string>( s, std::string( view.data(), view.size() ) );
        }
    }
};

// --- Enum from string column ---

template<typename ArrowStringArrayT>
class EnumFromStringReader final : public FieldReader
{
public:
    EnumFromStringReader( const std::string & columnName, const StructFieldPtr & field )
        : FieldReader( columnName, field ),
          m_enumMeta( std::static_pointer_cast<const CspEnumType>( field -> type() ) -> meta() )
    {
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ArrowStringArrayT &>( *m_column );
        if( typed.IsValid( row ) )
        {
            auto view = typed.GetView( row );
            // fromString requires null-terminated C string; view may not be null-terminated
            m_tmpStr.assign( view.data(), view.size() );
            m_field -> setValue<CspEnum>( s, m_enumMeta -> fromString( m_tmpStr.c_str() ) );
        }
    }

private:
    std::shared_ptr<const CspEnumMeta> m_enumMeta;
    mutable std::string                m_tmpStr;   // reused buffer to avoid per-row allocation
};

// --- Binary / bytes readers ---

template<typename ArrowBinaryArrayT>
class BytesReader final : public FieldReader
{
public:
    using FieldReader::FieldReader;

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ArrowBinaryArrayT &>( *m_column );
        if( typed.IsValid( row ) )
        {
            auto view = typed.GetView( row );
            m_field -> setValue<std::string>( s, std::string( view.data(), view.size() ) );
        }
    }
};

// --- Timestamp -> DateTime ---

class TimestampReader final : public FieldReader
{
public:
    TimestampReader( const std::string & columnName, const StructFieldPtr & field, int64_t multiplier )
        : FieldReader( columnName, field ), m_multiplier( multiplier )
    {
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::TimestampArray &>( *m_column );
        if( typed.IsValid( row ) )
            m_field -> setValue<DateTime>( s, DateTime::fromNanoseconds( typed.Value( row ) * m_multiplier ) );
    }

private:
    int64_t m_multiplier;
};

// --- Duration -> TimeDelta ---

class DurationReader final : public FieldReader
{
public:
    DurationReader( const std::string & columnName, const StructFieldPtr & field, int64_t multiplier )
        : FieldReader( columnName, field ), m_multiplier( multiplier )
    {
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::DurationArray &>( *m_column );
        if( typed.IsValid( row ) )
            m_field -> setValue<TimeDelta>( s, TimeDelta::fromNanoseconds( typed.Value( row ) * m_multiplier ) );
    }

private:
    int64_t m_multiplier;
};

// --- Date32 -> Date ---

class Date32Reader final : public FieldReader
{
public:
    using FieldReader::FieldReader;

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::Date32Array &>( *m_column );
        if( typed.IsValid( row ) )
        {
            int64_t nanos = static_cast<int64_t>( typed.Value( row ) ) * csp::NANOS_PER_DAY;
            m_field -> setValue<Date>( s, DateTime::fromNanoseconds( nanos ).date() );
        }
    }
};

// --- Date64 -> Date ---

class Date64Reader final : public FieldReader
{
public:
    using FieldReader::FieldReader;

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::Date64Array &>( *m_column );
        if( typed.IsValid( row ) )
        {
            int64_t nanos = typed.Value( row ) * csp::NANOS_PER_MILLISECOND;
            m_field -> setValue<Date>( s, DateTime::fromNanoseconds( nanos ).date() );
        }
    }
};

// --- Time32 -> Time ---

class Time32Reader final : public FieldReader
{
public:
    Time32Reader( const std::string & columnName, const StructFieldPtr & field, int64_t multiplier )
        : FieldReader( columnName, field ), m_multiplier( multiplier )
    {
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::Time32Array &>( *m_column );
        if( typed.IsValid( row ) )
            m_field -> setValue<Time>( s, Time::fromNanoseconds( static_cast<int64_t>( typed.Value( row ) ) * m_multiplier ) );
    }

private:
    int64_t m_multiplier;
};

// --- Time64 -> Time ---

class Time64Reader final : public FieldReader
{
public:
    Time64Reader( const std::string & columnName, const StructFieldPtr & field, int64_t multiplier )
        : FieldReader( columnName, field ), m_multiplier( multiplier )
    {
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::Time64Array &>( *m_column );
        if( typed.IsValid( row ) )
            m_field -> setValue<Time>( s, Time::fromNanoseconds( typed.Value( row ) * m_multiplier ) );
    }

private:
    int64_t m_multiplier;
};

// --- Dictionary-encoded string ---

class DictStringReader final : public FieldReader
{
public:
    using FieldReader::FieldReader;

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::DictionaryArray &>( *m_column );
        // Cache the dictionary pointer on first row of each batch (bindColumn resets m_row to 0)
        if( row == 0 )
            m_dict = &static_cast<const ::arrow::StringArray &>( *typed.dictionary() );
        if( typed.IsValid( row ) )
        {
            auto view = m_dict -> GetView( typed.GetValueIndex( row ) );
            m_field -> setValue<std::string>( s, std::string( view.data(), view.size() ) );
        }
    }

private:
    const ::arrow::StringArray * m_dict = nullptr;
};

// --- Dictionary-encoded enum ---

class DictEnumReader final : public FieldReader
{
public:
    DictEnumReader( const std::string & columnName, const StructFieldPtr & field )
        : FieldReader( columnName, field ),
          m_enumMeta( std::static_pointer_cast<const CspEnumType>( field -> type() ) -> meta() )
    {
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::DictionaryArray &>( *m_column );
        if( row == 0 )
            m_dict = &static_cast<const ::arrow::StringArray &>( *typed.dictionary() );
        if( typed.IsValid( row ) )
        {
            auto view = m_dict -> GetView( typed.GetValueIndex( row ) );
            m_tmpStr.assign( view.data(), view.size() );
            m_field -> setValue<CspEnum>( s, m_enumMeta -> fromString( m_tmpStr.c_str() ) );
        }
    }

private:
    std::shared_ptr<const CspEnumMeta> m_enumMeta;
    const ::arrow::StringArray *       m_dict = nullptr;
    mutable std::string                m_tmpStr;
};

// --- Nested struct (recursive) ---
// Children are bound to their child arrays on row 0 (i.e. after each bindColumn reset).
// readNext()/skipNext() keep child row counters in sync with the parent.

class NestedStructReader final : public FieldReader
{
public:
    NestedStructReader( const std::string & columnName, const StructFieldPtr & field,
                        const std::shared_ptr<::arrow::DataType> & arrowType )
        : FieldReader( columnName, field )
    {
        m_nestedMeta = std::static_pointer_cast<const CspStructType>( field -> type() ) -> meta();
        auto structType = std::static_pointer_cast<::arrow::StructType>( arrowType );

        m_childReaders.reserve( structType -> num_fields() );
        for( int i = 0; i < structType -> num_fields(); ++i )
        {
            auto childArrowField = structType -> field( i );
            auto childStructField = m_nestedMeta -> field( childArrowField -> name() );
            if( !childStructField )
                CSP_THROW( RuntimeException, "Nested arrow struct field '" << childArrowField -> name()
                                              << "' not found on CSP struct type '" << m_nestedMeta -> name() << "'" );

            m_childIndices.push_back( i );
            m_childReaders.push_back( createFieldReader( childArrowField, childStructField ) );
        }
    }

protected:
    void doReadNext( int64_t row, Struct * s ) override
    {
        auto & typed = static_cast<const ::arrow::StructArray &>( *m_column );

        // Bind children to their child arrays on the first row of each batch.
        // bindColumn() resets m_row to 0, so row == 0 fires once per bind.
        if( row == 0 )
        {
            for( size_t i = 0; i < m_childReaders.size(); ++i )
                m_childReaders[i] -> bindColumn( typed.field( m_childIndices[i] ).get() );
        }

        if( typed.IsValid( row ) )
        {
            StructPtr nested = m_nestedMeta -> create();
            for( auto & child : m_childReaders )
                child -> readNext( nested.get() );
            m_field -> setValue<StructPtr>( s, std::move( nested ) );
        }
        else
        {
            // Parent is null — advance children to keep row counters in sync
            for( auto & child : m_childReaders )
                child -> skipNext();
        }
    }

private:
    std::shared_ptr<const StructMeta>         m_nestedMeta;
    std::vector<int>                          m_childIndices;
    std::vector<std::unique_ptr<FieldReader>> m_childReaders;
};

} // anonymous namespace

std::unique_ptr<FieldReader> createFieldReader(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const StructFieldPtr & structField )
{
    bool isEnum = structField -> type() -> type() == CspType::Type::ENUM;
    auto typeId = arrowField -> type() -> id();
    auto & name = arrowField -> name();

    switch( typeId )
    {
        // --- Numeric natives ---
        case ::arrow::Type::BOOL:
            return std::make_unique<PrimitiveReader<bool, ::arrow::BooleanArray>>( name, structField );

        case ::arrow::Type::INT8:
            return std::make_unique<PrimitiveReader<int8_t, ::arrow::Int8Array>>( name, structField );
        case ::arrow::Type::INT16:
            return std::make_unique<PrimitiveReader<int16_t, ::arrow::Int16Array>>( name, structField );
        case ::arrow::Type::INT32:
            return std::make_unique<PrimitiveReader<int32_t, ::arrow::Int32Array>>( name, structField );
        case ::arrow::Type::INT64:
            return std::make_unique<PrimitiveReader<int64_t, ::arrow::Int64Array>>( name, structField );

        case ::arrow::Type::UINT8:
            return std::make_unique<PrimitiveReader<uint8_t, ::arrow::UInt8Array>>( name, structField );
        case ::arrow::Type::UINT16:
            return std::make_unique<PrimitiveReader<uint16_t, ::arrow::UInt16Array>>( name, structField );
        case ::arrow::Type::UINT32:
            return std::make_unique<PrimitiveReader<uint32_t, ::arrow::UInt32Array>>( name, structField );
        case ::arrow::Type::UINT64:
            return std::make_unique<PrimitiveReader<uint64_t, ::arrow::UInt64Array>>( name, structField );

        case ::arrow::Type::HALF_FLOAT:
            return std::make_unique<HalfFloatReader>( name, structField );
        case ::arrow::Type::FLOAT:
            return std::make_unique<PrimitiveReader<double, ::arrow::FloatArray>>( name, structField );
        case ::arrow::Type::DOUBLE:
            return std::make_unique<PrimitiveReader<double, ::arrow::DoubleArray>>( name, structField );

        // --- String ---
        case ::arrow::Type::STRING:
            if( isEnum )
                return std::make_unique<EnumFromStringReader<::arrow::StringArray>>( name, structField );
            return std::make_unique<StringReader<::arrow::StringArray>>( name, structField );

        case ::arrow::Type::LARGE_STRING:
            if( isEnum )
                return std::make_unique<EnumFromStringReader<::arrow::LargeStringArray>>( name, structField );
            return std::make_unique<StringReader<::arrow::LargeStringArray>>( name, structField );

        // --- Binary / bytes ---
        case ::arrow::Type::BINARY:
            return std::make_unique<BytesReader<::arrow::BinaryArray>>( name, structField );
        case ::arrow::Type::LARGE_BINARY:
            return std::make_unique<BytesReader<::arrow::LargeBinaryArray>>( name, structField );
        case ::arrow::Type::FIXED_SIZE_BINARY:
            return std::make_unique<BytesReader<::arrow::FixedSizeBinaryArray>>( name, structField );

        // --- Timestamp -> DateTime ---
        case ::arrow::Type::TIMESTAMP:
        {
            auto tsType = std::static_pointer_cast<::arrow::TimestampType>( arrowField -> type() );
            return std::make_unique<TimestampReader>( name, structField, timeUnitMultiplier( tsType -> unit() ) );
        }

        // --- Duration -> TimeDelta ---
        case ::arrow::Type::DURATION:
        {
            auto durType = std::static_pointer_cast<::arrow::DurationType>( arrowField -> type() );
            return std::make_unique<DurationReader>( name, structField, timeUnitMultiplier( durType -> unit() ) );
        }

        // --- Date ---
        case ::arrow::Type::DATE32:
            return std::make_unique<Date32Reader>( name, structField );
        case ::arrow::Type::DATE64:
            return std::make_unique<Date64Reader>( name, structField );

        // --- Time ---
        case ::arrow::Type::TIME32:
        {
            auto t32Type = std::static_pointer_cast<::arrow::Time32Type>( arrowField -> type() );
            return std::make_unique<Time32Reader>( name, structField, timeUnitMultiplier( t32Type -> unit() ) );
        }
        case ::arrow::Type::TIME64:
        {
            auto t64Type = std::static_pointer_cast<::arrow::Time64Type>( arrowField -> type() );
            return std::make_unique<Time64Reader>( name, structField, timeUnitMultiplier( t64Type -> unit() ) );
        }

        // --- Dictionary-encoded ---
        case ::arrow::Type::DICTIONARY:
        {
            auto dictType = std::static_pointer_cast<::arrow::DictionaryType>( arrowField -> type() );
            if( dictType -> value_type() -> id() != ::arrow::Type::STRING )
                CSP_THROW( TypeError, "Unsupported dictionary value type " << dictType -> value_type() -> ToString()
                                       << " for column '" << name << "'; only string dictionaries supported" );
            if( isEnum )
                return std::make_unique<DictEnumReader>( name, structField );
            return std::make_unique<DictStringReader>( name, structField );
        }

        // --- Nested struct ---
        case ::arrow::Type::STRUCT:
            return std::make_unique<NestedStructReader>( name, structField, arrowField -> type() );

        default:
            CSP_THROW( TypeError, "Unsupported arrow type " << arrowField -> type() -> ToString()
                                   << " for column '" << name << "'" );
    }
}

}
