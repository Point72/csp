// Typed ColumnDispatcher implementations and factory.

#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <csp/adapters/arrow/ArrowTypeVisitor.h>
#include <csp/adapters/utils/ValueDispatcher.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/CspType.h>
#include <csp/engine/PartialSwitchCspType.h>

#include <arrow/array.h>
#include <arrow/util/float16.h>

namespace csp::adapters::arrow
{

namespace
{

// Lightweight reader for the inlined (fast) path: just a cached typed pointer + extract fn.
// No FieldReader overhead — only 8 + sizeof(ExtractFn) bytes.
template<typename ArrowArrayT, typename ValueT, typename ExtractFn>
struct InlineReader
{
    InlineReader( ExtractFn fn ) : m_extractFn( std::move( fn ) ) {}

    void readDirect( int64_t row, std::optional<ValueT> & out )
    {
        if( m_typedColumn -> IsValid( row ) )
            out = m_extractFn( *m_typedColumn, row );
        else
            out.reset();
    }

    void bindColumn( const ::arrow::Array * col )
    {
        m_typedColumn = static_cast<const ArrowArrayT *>( col );
    }

    const ArrowArrayT * m_typedColumn = nullptr;
    ExtractFn           m_extractFn;
};

// Wrapper for runtime-polymorphic FieldReader (used by STRUCT, DICTIONARY, LIST).
template<typename ValueType>
struct ErasedReader
{
    explicit ErasedReader( std::unique_ptr<FieldReader> reader )
        : m_reader( std::move( reader ) ) {}

    void readDirect( int64_t row, std::optional<ValueType> & out )
    {
        m_reader -> readValueAt( row, &out );
    }
    void bindColumn( const ::arrow::Array * col ) { m_reader -> bindColumn( col ); }

    std::unique_ptr<FieldReader> m_reader;
};


// TypedColumnDispatcher: holds a ConcreteReader by value.
// For primitive types, ConcreteReader is InlineReader (tiny, fast).
// For complex types, ConcreteReader is ErasedReader (virtual fallback).
template<typename ValueType, typename ConcreteReader>
class TypedColumnDispatcher final : public ColumnDispatcher
{
    using Dispatcher = utils::ValueDispatcher<const ValueType &>;

public:
    TypedColumnDispatcher( std::string name, ConcreteReader reader,
                           ::arrow::Type::type arrowTypeId )
        : ColumnDispatcher( std::move( name ), arrowTypeId ),
          m_reader( std::move( reader ) )
    {
    }

    void readValueAt( int64_t row ) override
    {
        m_reader.readDirect( row, m_value );
    }

    void bindColumn( const ::arrow::Array * column ) override
    {
        m_reader.bindColumn( column );
    }

    void dispatchValue( const utils::Symbol * symbol ) override
    {
        if( m_value.has_value() )
            m_dispatcher.dispatch( &m_value.value(), symbol );
        else
            m_dispatcher.dispatch( nullptr, symbol );
    }

    void * getCurValueUntyped() override { return &m_value; }

    void addSubscriber( ManagedSimInputAdapter * adapter,
                        std::optional<utils::Symbol> symbol ) override
    {
        if constexpr( std::is_same_v<ValueType, std::string> )
        {
            if( adapter -> type() -> type() == CspType::Type::ENUM )
            {
                auto enumMetaPtr = static_cast<const CspEnumType *>( adapter -> type() ) -> meta();
                auto callback = typename Dispatcher::SubscriberType(
                    [adapter, enumMetaPtr]( const std::string * val )
                    {
                        if( val )
                            adapter -> pushTick<CspEnum>( enumMetaPtr -> fromString( val -> c_str() ) );
                        else
                            adapter -> pushNullTick<CspEnum>();
                    } );
                m_dispatcher.addSubscriber( callback, symbol );
                return;
            }
        }
        addSubscriberImpl( adapter, symbol );
    }

private:
    void addSubscriberImpl( ManagedSimInputAdapter * adapter,
                            std::optional<utils::Symbol> symbol )
    {
        using Switch = ConstructibleTypeSwitch<ValueType>;
        try
        {
            auto callback = Switch::invoke( adapter -> type(), [adapter]( auto tag )
            {
                return typename Dispatcher::SubscriberType(
                    [adapter]( const ValueType * val )
                    {
                        if( val )
                            adapter -> pushTick<typename decltype( tag )::type>( *val );
                        else
                            adapter -> pushNullTick<typename decltype( tag )::type>();
                    } );
            } );
            m_dispatcher.addSubscriber( callback, symbol );
        }
        catch( UnsupportedSwitchType & )
        {
            CSP_THROW( TypeError, "Unsupported subscriber type "
                                   << adapter -> type() -> type().asCString()
                                   << " for column '" << m_columnName << "'" );
        }
    }

    ConcreteReader          m_reader;
    std::optional<ValueType> m_value;
    Dispatcher              m_dispatcher;
};


// Helper: create a TypedColumnDispatcher with an InlineReader (fast path).
template<typename ValueType, typename ArrowArrayT, typename ExtractFn>
std::unique_ptr<ColumnDispatcher> makeInlinedDispatcher(
    const std::string & name, ::arrow::Type::type typeId, ExtractFn extractFn )
{
    using Reader = InlineReader<ArrowArrayT, ValueType, ExtractFn>;
    Reader reader( std::move( extractFn ) );
    return std::make_unique<TypedColumnDispatcher<ValueType, Reader>>(
        name, std::move( reader ), typeId );
}

// Helper: create a TypedColumnDispatcher with an ErasedReader (virtual fallback).
template<typename ValueType>
std::unique_ptr<ColumnDispatcher> makeErasedDispatcher(
    const std::string & name, std::unique_ptr<FieldReader> reader, ::arrow::Type::type typeId )
{
    ErasedReader<ValueType> erased( std::move( reader ) );
    return std::make_unique<TypedColumnDispatcher<ValueType, ErasedReader<ValueType>>>(
        name, std::move( erased ), typeId );
}

} // anonymous namespace


std::unique_ptr<ColumnDispatcher> createColumnDispatcher(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const std::shared_ptr<const StructMeta> & structMeta )
{
    auto typeId = arrowField -> type() -> id();
    auto & name = arrowField -> name();

    if( typeId == ::arrow::Type::STRUCT && !structMeta )
        return nullptr;

    // Primitive types: use InlineReader (cached typed pointer + extract lambda).
    switch( typeId )
    {
    case ::arrow::Type::INT8:
        return makeInlinedDispatcher<int8_t, ::arrow::Int8Array>( name, typeId,
            []( const ::arrow::Int8Array & arr, int64_t i ) -> int8_t { return arr.Value( i ); } );
    case ::arrow::Type::INT16:
        return makeInlinedDispatcher<int16_t, ::arrow::Int16Array>( name, typeId,
            []( const ::arrow::Int16Array & arr, int64_t i ) -> int16_t { return arr.Value( i ); } );
    case ::arrow::Type::INT32:
        return makeInlinedDispatcher<int32_t, ::arrow::Int32Array>( name, typeId,
            []( const ::arrow::Int32Array & arr, int64_t i ) -> int32_t { return arr.Value( i ); } );
    case ::arrow::Type::INT64:
        return makeInlinedDispatcher<int64_t, ::arrow::Int64Array>( name, typeId,
            []( const ::arrow::Int64Array & arr, int64_t i ) -> int64_t { return arr.Value( i ); } );
    case ::arrow::Type::UINT8:
        return makeInlinedDispatcher<uint8_t, ::arrow::UInt8Array>( name, typeId,
            []( const ::arrow::UInt8Array & arr, int64_t i ) -> uint8_t { return arr.Value( i ); } );
    case ::arrow::Type::UINT16:
        return makeInlinedDispatcher<uint16_t, ::arrow::UInt16Array>( name, typeId,
            []( const ::arrow::UInt16Array & arr, int64_t i ) -> uint16_t { return arr.Value( i ); } );
    case ::arrow::Type::UINT32:
        return makeInlinedDispatcher<uint32_t, ::arrow::UInt32Array>( name, typeId,
            []( const ::arrow::UInt32Array & arr, int64_t i ) -> uint32_t { return arr.Value( i ); } );
    case ::arrow::Type::UINT64:
        return makeInlinedDispatcher<uint64_t, ::arrow::UInt64Array>( name, typeId,
            []( const ::arrow::UInt64Array & arr, int64_t i ) -> uint64_t { return arr.Value( i ); } );
    case ::arrow::Type::FLOAT:
        return makeInlinedDispatcher<double, ::arrow::FloatArray>( name, typeId,
            []( const ::arrow::FloatArray & arr, int64_t i ) -> double { return static_cast<double>( arr.Value( i ) ); } );
    case ::arrow::Type::DOUBLE:
        return makeInlinedDispatcher<double, ::arrow::DoubleArray>( name, typeId,
            []( const ::arrow::DoubleArray & arr, int64_t i ) -> double { return arr.Value( i ); } );
    case ::arrow::Type::BOOL:
        return makeInlinedDispatcher<bool, ::arrow::BooleanArray>( name, typeId,
            []( const ::arrow::BooleanArray & arr, int64_t i ) -> bool { return arr.Value( i ); } );
    case ::arrow::Type::HALF_FLOAT:
        return makeInlinedDispatcher<double, ::arrow::HalfFloatArray>( name, typeId,
            []( const ::arrow::HalfFloatArray & arr, int64_t i ) -> double {
                return ::arrow::util::Float16::FromBits( arr.Value( i ) ).ToDouble();
            } );
    case ::arrow::Type::STRING:
        return makeInlinedDispatcher<std::string, ::arrow::StringArray>( name, typeId,
            []( const ::arrow::StringArray & arr, int64_t i ) -> std::string {
                auto view = arr.GetView( i );
                return std::string( view.data(), view.size() );
            } );
    case ::arrow::Type::LARGE_STRING:
        return makeInlinedDispatcher<std::string, ::arrow::LargeStringArray>( name, typeId,
            []( const ::arrow::LargeStringArray & arr, int64_t i ) -> std::string {
                auto view = arr.GetView( i );
                return std::string( view.data(), view.size() );
            } );
    case ::arrow::Type::BINARY:
        return makeInlinedDispatcher<std::string, ::arrow::BinaryArray>( name, typeId,
            []( const ::arrow::BinaryArray & arr, int64_t i ) -> std::string {
                auto view = arr.GetView( i );
                return std::string( view.data(), view.size() );
            } );
    case ::arrow::Type::LARGE_BINARY:
        return makeInlinedDispatcher<std::string, ::arrow::LargeBinaryArray>( name, typeId,
            []( const ::arrow::LargeBinaryArray & arr, int64_t i ) -> std::string {
                auto view = arr.GetView( i );
                return std::string( view.data(), view.size() );
            } );
    case ::arrow::Type::FIXED_SIZE_BINARY:
        return makeInlinedDispatcher<std::string, ::arrow::FixedSizeBinaryArray>( name, typeId,
            []( const ::arrow::FixedSizeBinaryArray & arr, int64_t i ) -> std::string {
                auto view = arr.GetView( i );
                return std::string( view.data(), view.size() );
            } );
    case ::arrow::Type::TIMESTAMP:
    {
        auto mult = timeUnitMultiplier(
            std::static_pointer_cast<::arrow::TimestampType>( arrowField -> type() ) -> unit() );
        return makeInlinedDispatcher<DateTime, ::arrow::TimestampArray>( name, typeId,
            [mult]( const ::arrow::TimestampArray & arr, int64_t i ) -> DateTime {
                return DateTime::fromNanoseconds( arr.Value( i ) * mult );
            } );
    }
    case ::arrow::Type::DURATION:
    {
        auto mult = timeUnitMultiplier(
            std::static_pointer_cast<::arrow::DurationType>( arrowField -> type() ) -> unit() );
        return makeInlinedDispatcher<TimeDelta, ::arrow::DurationArray>( name, typeId,
            [mult]( const ::arrow::DurationArray & arr, int64_t i ) -> TimeDelta {
                return TimeDelta::fromNanoseconds( arr.Value( i ) * mult );
            } );
    }
    case ::arrow::Type::DATE32:
        return makeInlinedDispatcher<Date, ::arrow::Date32Array>( name, typeId,
            []( const ::arrow::Date32Array & arr, int64_t i ) -> Date {
                return DateTime::fromNanoseconds( static_cast<int64_t>( arr.Value( i ) ) * NANOS_PER_DAY ).date();
            } );
    case ::arrow::Type::DATE64:
        return makeInlinedDispatcher<Date, ::arrow::Date64Array>( name, typeId,
            []( const ::arrow::Date64Array & arr, int64_t i ) -> Date {
                return DateTime::fromNanoseconds( arr.Value( i ) * NANOS_PER_MILLISECOND ).date();
            } );
    case ::arrow::Type::TIME32:
    {
        auto mult = timeUnitMultiplier(
            std::static_pointer_cast<::arrow::Time32Type>( arrowField -> type() ) -> unit() );
        return makeInlinedDispatcher<Time, ::arrow::Time32Array>( name, typeId,
            [mult]( const ::arrow::Time32Array & arr, int64_t i ) -> Time {
                return Time::fromNanoseconds( static_cast<int64_t>( arr.Value( i ) ) * mult );
            } );
    }
    case ::arrow::Type::TIME64:
    {
        auto mult = timeUnitMultiplier(
            std::static_pointer_cast<::arrow::Time64Type>( arrowField -> type() ) -> unit() );
        return makeInlinedDispatcher<Time, ::arrow::Time64Array>( name, typeId,
            [mult]( const ::arrow::Time64Array & arr, int64_t i ) -> Time {
                return Time::fromNanoseconds( arr.Value( i ) * mult );
            } );
    }
    default:
        break;
    }

    // Fallback for complex types (DICTIONARY, STRUCT, LIST): use ErasedReader.
    auto fieldReader = createFieldReader( arrowField, nullptr, structMeta );
    if( !fieldReader )
        return nullptr;

    return visitArrowValueType( typeId,
        [&]( auto tag ) -> std::unique_ptr<ColumnDispatcher>
        {
            using T = typename decltype( tag )::type;
            return makeErasedDispatcher<T>( name, std::move( fieldReader ), typeId );
        },
        []() -> std::unique_ptr<ColumnDispatcher> { return nullptr; } );
}

} // namespace csp::adapters::arrow
