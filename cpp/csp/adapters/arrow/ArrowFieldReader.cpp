// Concrete FieldReader implementations for Arrow types.

#include <csp/adapters/arrow/ArrowFieldReader.h>
#include <csp/engine/CspType.h>
#include <csp/engine/CspEnum.h>

#include <arrow/array.h>
#include <arrow/type.h>
#include <arrow/util/float16.h>

namespace csp::adapters::arrow
{

static ListFieldReaderFactory s_listFieldReaderFactory;

void registerListFieldReaderFactory( ListFieldReaderFactory factory )
{
    s_listFieldReaderFactory = std::move( factory );
}

namespace
{

// Columnar bulk-read helper: dispatches fn(arr, row, struct*) for each row,
// skipping nulls when null_count > 0.
template<typename ArrowArrayT, typename Fn>
void readColumn( const ArrowArrayT & typed, std::vector<StructPtr> & structs, int64_t numRows, Fn && fn )
{
    if( typed.null_count() == 0 )
        for( int64_t i = 0; i < numRows; ++i )
            fn( typed, i, structs[i].get() );
    else
        for( int64_t i = 0; i < numRows; ++i )
            if( typed.IsValid( i ) )
                fn( typed, i, structs[i].get() );
}

template<typename ArrowArrayT, typename ValueT, typename ExtractFn>
class LambdaReader final : public TypedFieldReader<ValueT>
{
    using Base = TypedFieldReader<ValueT>;
public:
    LambdaReader( const std::string & columnName, const StructFieldPtr & field,
                  ExtractFn extractFn )
        : Base( columnName, field ), m_extractFn( std::move( extractFn ) ) {}

    void readAll( std::vector<StructPtr> & structs, int64_t numRows ) override
    {
        auto & typed = static_cast<const ArrowArrayT &>( *this -> m_column );
        readColumn( typed, structs, numRows, [this]( auto & arr, int64_t i, Struct * s ) {
            this -> m_field -> template setValue<ValueT>( s, m_extractFn( arr, i ) );
        } );
        this -> m_row = numRows;
    }

protected:
    bool doExtract( int64_t row, ValueT & out ) override
    {
        auto & typed = static_cast<const ArrowArrayT &>( *this -> m_column );
        if( typed.IsValid( row ) )
        {
            out = m_extractFn( typed, row );
            return true;
        }
        return false;
    }

private:
    ExtractFn  m_extractFn;
};

template<typename ArrowArrayT, typename ValueT, typename ExtractFn>
std::unique_ptr<FieldReader> makeReader( const std::string & name, const StructFieldPtr & field,
                                         ExtractFn && extractFn )
{
    return std::make_unique<LambdaReader<ArrowArrayT, ValueT,
                                         std::decay_t<ExtractFn>>>(
        name, field, std::forward<ExtractFn>( extractFn ) );
}

template<typename CspT, typename ArrowArrayT>
std::unique_ptr<FieldReader> makePrimitiveReader( const std::string & name, const StructFieldPtr & f )
{
    return makeReader<ArrowArrayT, CspT>( name, f,
        []( auto & arr, int64_t i ) -> CspT {
            return static_cast<CspT>( arr.Value( i ) );
        } );
}

template<typename ArrowArrayT>
std::unique_ptr<FieldReader> makeStringReader( const std::string & name, const StructFieldPtr & f )
{
    return makeReader<ArrowArrayT, std::string>( name, f,
        []( auto & arr, int64_t i ) -> std::string {
            auto view = arr.GetView( i );
            return std::string( view.data(), view.size() );
        } );
}

template<typename CspT, typename ArrowArrayT>
std::unique_ptr<FieldReader> makeNanosReader( const std::string & name, const StructFieldPtr & f, int64_t mult )
{
    return makeReader<ArrowArrayT, CspT>( name, f,
        [mult]( auto & arr, int64_t i ) -> CspT {
            return CspT::fromNanoseconds( static_cast<int64_t>( arr.Value( i ) ) * mult );
        } );
}


template<typename ArrowStringArrayT>
class EnumFromStringReader final : public TypedFieldReader<CspEnum>
{
    using Base = TypedFieldReader<CspEnum>;
public:
    EnumFromStringReader( const std::string & columnName, const StructFieldPtr & field )
        : Base( columnName, field ),
          m_enumMeta( std::static_pointer_cast<const CspEnumType>( field -> type() ) -> meta() ) {}

    void readAll( std::vector<StructPtr> & structs, int64_t numRows ) override
    {
        auto & typed = static_cast<const ArrowStringArrayT &>( *this -> m_column );
        readColumn( typed, structs, numRows, [this]( auto & arr, int64_t i, Struct * s ) {
            auto view = arr.GetView( i );
            m_tmpStr.assign( view.data(), view.size() );
            this -> m_field -> template setValue<CspEnum>( s, m_enumMeta -> fromString( m_tmpStr.c_str() ) );
        } );
        this -> m_row = numRows;
    }

protected:
    bool doExtract( int64_t row, CspEnum & out ) override
    {
        auto & typed = static_cast<const ArrowStringArrayT &>( *this -> m_column );
        if( typed.IsValid( row ) )
        {
            auto view = typed.GetView( row );
            m_tmpStr.assign( view.data(), view.size() );
            out = m_enumMeta -> fromString( m_tmpStr.c_str() );
            return true;
        }
        return false;
    }

private:
    std::shared_ptr<const CspEnumMeta> m_enumMeta;
    mutable std::string                m_tmpStr;
};


template<typename ValueT, typename ConvertFn>
class DictReader final : public TypedFieldReader<ValueT>
{
    using Base = TypedFieldReader<ValueT>;
public:
    DictReader( const std::string & columnName, const StructFieldPtr & field, ConvertFn convert )
        : Base( columnName, field ), m_convert( std::move( convert ) ) {}

    void onBind() override
    {
        auto & typed = static_cast<const ::arrow::DictionaryArray &>( *this -> m_column );
        m_dict = &static_cast<const ::arrow::StringArray &>( *typed.dictionary() );
    }

    void readAll( std::vector<StructPtr> & structs, int64_t numRows ) override
    {
        auto & typed = static_cast<const ::arrow::DictionaryArray &>( *this -> m_column );
        const auto * dict = &static_cast<const ::arrow::StringArray &>( *typed.dictionary() );
        readColumn( typed, structs, numRows, [this, dict]( auto & arr, int64_t i, Struct * s ) {
            auto view = dict -> GetView( arr.GetValueIndex( i ) );
            this -> m_field -> template setValue<ValueT>( s, m_convert( view ) );
        } );
        this -> m_row = numRows;
    }

protected:
    bool doExtract( int64_t row, ValueT & out ) override
    {
        auto & typed = static_cast<const ::arrow::DictionaryArray &>( *this -> m_column );
        if( typed.IsValid( row ) )
        {
            auto view = m_dict -> GetView( typed.GetValueIndex( row ) );
            out = m_convert( view );
            return true;
        }
        return false;
    }

private:
    ConvertFn                    m_convert;
    const ::arrow::StringArray * m_dict = nullptr;
};

template<typename ValueT, typename ConvertFn>
std::unique_ptr<FieldReader> makeDictReader( const std::string & name, const StructFieldPtr & field,
                                              ConvertFn && convert )
{
    return std::make_unique<DictReader<ValueT, std::decay_t<ConvertFn>>>(
        name, field, std::forward<ConvertFn>( convert ) );
}


class NestedStructReader final : public TypedFieldReader<StructPtr>
{
    using Base = TypedFieldReader<StructPtr>;
public:
    NestedStructReader( const std::string & columnName, const StructFieldPtr & field,
                        const std::shared_ptr<::arrow::DataType> & arrowType,
                        std::shared_ptr<const StructMeta> explicitMeta = nullptr )
        : Base( columnName, field )
    {
        m_nestedMeta = explicitMeta ? std::move( explicitMeta )
                                    : std::static_pointer_cast<const CspStructType>( field -> type() ) -> meta();
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

    void onBind() override
    {
        auto & typed = static_cast<const ::arrow::StructArray &>( *m_column );
        for( size_t i = 0; i < m_childReaders.size(); ++i )
            m_childReaders[i] -> bindColumn( typed.field( m_childIndices[i] ).get() );
    }

    void readAll( std::vector<StructPtr> & structs, int64_t numRows ) override
    {
        auto & typed = static_cast<const ::arrow::StructArray &>( *m_column );

        if( typed.null_count() == 0 )
        {
            // Pre-allocate nested structs and let children use their columnar readAll paths
            std::vector<StructPtr> nested( numRows );
            for( int64_t i = 0; i < numRows; ++i )
                nested[i] = m_nestedMeta -> create();
            for( auto & child : m_childReaders )
                child -> readAll( nested, numRows );
            for( int64_t row = 0; row < numRows; ++row )
                m_field -> setValue<StructPtr>( structs[row].get(), std::move( nested[row] ) );
        }
        else
        {
            for( int64_t row = 0; row < numRows; ++row )
            {
                if( typed.IsValid( row ) )
                {
                    StructPtr nested = m_nestedMeta -> create();
                    for( auto & child : m_childReaders )
                        child -> readNext( nested.get() );
                    m_field -> setValue<StructPtr>( structs[row].get(), std::move( nested ) );
                }
                else
                {
                    for( auto & child : m_childReaders )
                        child -> skipNext();
                }
            }
        }
        m_row = numRows;
    }

protected:
    void skipNext() override
    {
        for( auto & child : m_childReaders )
            child -> skipNext();
        ++m_row;
    }

    bool doExtract( int64_t row, StructPtr & out ) override
    {
        auto & typed = static_cast<const ::arrow::StructArray &>( *m_column );
        if( row == 0 )
            for( size_t i = 0; i < m_childReaders.size(); ++i )
                m_childReaders[i] -> bindColumn( typed.field( m_childIndices[i] ).get() );

        if( typed.IsValid( row ) )
        {
            StructPtr nested = m_nestedMeta -> create();
            for( auto & child : m_childReaders )
                child -> readNext( nested.get() );
            out = std::move( nested );
            return true;
        }
        else
        {
            for( auto & child : m_childReaders )
                child -> skipNext();
            return false;
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
    const StructFieldPtr & structField,
    const std::shared_ptr<const StructMeta> & structMeta )
{
    bool isEnum = structField && structField -> type() -> type() == CspType::Type::ENUM;
    auto typeId = arrowField -> type() -> id();
    auto & name = arrowField -> name();
    auto & f    = structField;

    switch( typeId )
    {
        case ::arrow::Type::BOOL:   return makePrimitiveReader<bool,     ::arrow::BooleanArray>( name, f );
        case ::arrow::Type::INT8:   return makePrimitiveReader<int8_t,   ::arrow::Int8Array>( name, f );
        case ::arrow::Type::INT16:  return makePrimitiveReader<int16_t,  ::arrow::Int16Array>( name, f );
        case ::arrow::Type::INT32:  return makePrimitiveReader<int32_t,  ::arrow::Int32Array>( name, f );
        case ::arrow::Type::INT64:  return makePrimitiveReader<int64_t,  ::arrow::Int64Array>( name, f );
        case ::arrow::Type::UINT8:  return makePrimitiveReader<uint8_t,  ::arrow::UInt8Array>( name, f );
        case ::arrow::Type::UINT16: return makePrimitiveReader<uint16_t, ::arrow::UInt16Array>( name, f );
        case ::arrow::Type::UINT32: return makePrimitiveReader<uint32_t, ::arrow::UInt32Array>( name, f );
        case ::arrow::Type::UINT64: return makePrimitiveReader<uint64_t, ::arrow::UInt64Array>( name, f );
        case ::arrow::Type::FLOAT:  return makePrimitiveReader<double,   ::arrow::FloatArray>( name, f );
        case ::arrow::Type::DOUBLE: return makePrimitiveReader<double,   ::arrow::DoubleArray>( name, f );

        case ::arrow::Type::HALF_FLOAT:
            return makeReader<::arrow::HalfFloatArray, double>( name, f,
                []( auto & arr, int64_t i ) -> double {
                    return ::arrow::util::Float16::FromBits( arr.Value( i ) ).ToDouble();
                } );

        case ::arrow::Type::STRING:
            if( isEnum ) return std::make_unique<EnumFromStringReader<::arrow::StringArray>>( name, f );
            return makeStringReader<::arrow::StringArray>( name, f );
        case ::arrow::Type::LARGE_STRING:
            if( isEnum ) return std::make_unique<EnumFromStringReader<::arrow::LargeStringArray>>( name, f );
            return makeStringReader<::arrow::LargeStringArray>( name, f );

        case ::arrow::Type::BINARY:            return makeStringReader<::arrow::BinaryArray>( name, f );
        case ::arrow::Type::LARGE_BINARY:      return makeStringReader<::arrow::LargeBinaryArray>( name, f );
        case ::arrow::Type::FIXED_SIZE_BINARY: return makeStringReader<::arrow::FixedSizeBinaryArray>( name, f );

        case ::arrow::Type::TIMESTAMP:
        {
            auto mult = timeUnitMultiplier( std::static_pointer_cast<::arrow::TimestampType>( arrowField -> type() ) -> unit() );
            return makeNanosReader<DateTime, ::arrow::TimestampArray>( name, f, mult );
        }

        case ::arrow::Type::DURATION:
        {
            auto mult = timeUnitMultiplier( std::static_pointer_cast<::arrow::DurationType>( arrowField -> type() ) -> unit() );
            return makeNanosReader<TimeDelta, ::arrow::DurationArray>( name, f, mult );
        }

        case ::arrow::Type::DATE32:
            return makeReader<::arrow::Date32Array, Date>( name, f,
                []( auto & arr, int64_t i ) -> Date {
                    return DateTime::fromNanoseconds( static_cast<int64_t>( arr.Value( i ) ) * csp::NANOS_PER_DAY ).date();
                } );
        case ::arrow::Type::DATE64:
            return makeReader<::arrow::Date64Array, Date>( name, f,
                []( auto & arr, int64_t i ) -> Date {
                    return DateTime::fromNanoseconds( arr.Value( i ) * csp::NANOS_PER_MILLISECOND ).date();
                } );

        case ::arrow::Type::TIME32:
        {
            auto mult = timeUnitMultiplier( std::static_pointer_cast<::arrow::Time32Type>( arrowField -> type() ) -> unit() );
            return makeNanosReader<Time, ::arrow::Time32Array>( name, f, mult );
        }
        case ::arrow::Type::TIME64:
        {
            auto mult = timeUnitMultiplier( std::static_pointer_cast<::arrow::Time64Type>( arrowField -> type() ) -> unit() );
            return makeNanosReader<Time, ::arrow::Time64Array>( name, f, mult );
        }

        case ::arrow::Type::DICTIONARY:
        {
            auto dictType = std::static_pointer_cast<::arrow::DictionaryType>( arrowField -> type() );
            if( dictType -> value_type() -> id() != ::arrow::Type::STRING )
                CSP_THROW( TypeError, "Unsupported dictionary value type " << dictType -> value_type() -> ToString()
                                       << " for column '" << name << "'; only string dictionaries supported" );
            if( isEnum )
            {
                auto enumMeta = std::static_pointer_cast<const CspEnumType>( f -> type() ) -> meta();
                return makeDictReader<CspEnum>( name, f,
                    [enumMeta, tmp = std::string{}]( auto view ) mutable -> CspEnum {
                        tmp.assign( view.data(), view.size() );
                        return enumMeta -> fromString( tmp.c_str() );
                    } );
            }
            return makeDictReader<std::string>( name, f,
                []( auto view ) -> std::string {
                    return std::string( view.data(), view.size() );
                } );
        }

        case ::arrow::Type::STRUCT:
            if( !f && !structMeta )
                return nullptr;  // no struct info available (ColumnDispatcher without meta)
            return std::make_unique<NestedStructReader>( name, f, arrowField -> type(), structMeta );

        case ::arrow::Type::LIST:
        case ::arrow::Type::LARGE_LIST:
            CSP_TRUE_OR_THROW_RUNTIME( s_listFieldReaderFactory,
                "List field reader factory not registered; ensure Python/numpy layer is initialized before reading list columns" );
            return s_listFieldReaderFactory( arrowField, structField );

        default:
            CSP_THROW( TypeError, "Unsupported arrow type " << arrowField -> type() -> ToString()
                                   << " for column '" << name << "'" );
    }
}

}
