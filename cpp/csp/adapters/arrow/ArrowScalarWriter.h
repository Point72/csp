#ifndef _IN_CSP_ADAPTERS_ARROW_ArrowScalarWriter_H
#define _IN_CSP_ADAPTERS_ARROW_ArrowScalarWriter_H

// Value-first Arrow column writer for CSP scalar types.
//
// ArrowScalarColumnWriter<CspT> owns one Arrow builder and knows how to convert+append a CSP value
// of type CspT to it. It is the single home of the per-type Arrow-builder choice and value
// conversion, shared by:
//   - the struct FieldWriter path (reads field->value<CspT>(s), then append()), and
//   - the scalar output column builder (append() the ticked value directly, no scratch struct).

#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/CspType.h>

#include <arrow/builder.h>
#include <arrow/type.h>
#include <arrow/type_traits.h>

#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>

namespace csp::adapters::arrow
{

template<class CspT>
class ArrowScalarColumnWriter
{
public:
    // cspType is only needed for the bytes-vs-utf8 distinction on STRING; isBytes carries it.
    explicit ArrowScalarColumnWriter( bool isBytes = false )
        : m_isBytes( isBytes )
    {
        if constexpr( std::is_same_v<CspT, DateTime> )
        {
            m_dataType = std::make_shared<::arrow::TimestampType>( ::arrow::TimeUnit::NANO, "UTC" );
            m_builder  = std::make_shared<::arrow::TimestampBuilder>( m_dataType, ::arrow::default_memory_pool() );
        }
        else if constexpr( std::is_same_v<CspT, TimeDelta> )
        {
            m_dataType = std::make_shared<::arrow::DurationType>( ::arrow::TimeUnit::NANO );
            m_builder  = std::make_shared<::arrow::DurationBuilder>( m_dataType, ::arrow::default_memory_pool() );
        }
        else if constexpr( std::is_same_v<CspT, Time> )
        {
            m_dataType = std::make_shared<::arrow::Time64Type>( ::arrow::TimeUnit::NANO );
            m_builder  = std::make_shared<::arrow::Time64Builder>( m_dataType, ::arrow::default_memory_pool() );
        }
        else if constexpr( std::is_same_v<CspT, Date> )
        {
            m_dataType = ::arrow::date32();
            m_builder  = std::make_shared<::arrow::Date32Builder>();
        }
        else if constexpr( std::is_same_v<CspT, std::string> )
        {
            if( isBytes ) { m_dataType = ::arrow::binary(); m_builder = std::make_shared<::arrow::BinaryBuilder>(); }
            else          { m_dataType = ::arrow::utf8();   m_builder = std::make_shared<::arrow::StringBuilder>(); }
        }
        else if constexpr( std::is_same_v<CspT, CspEnum> )
        {
            m_dataType = ::arrow::utf8();
            m_builder  = std::make_shared<::arrow::StringBuilder>();
        }
        else if constexpr( std::is_same_v<CspT, bool> )
        {
            m_dataType = ::arrow::boolean();
            m_builder  = std::make_shared<::arrow::BooleanBuilder>();
        }
        else  // numeric
        {
            using BuilderT = typename ::arrow::CTypeTraits<CspT>::BuilderType;
            m_dataType = ::arrow::CTypeTraits<CspT>::type_singleton();
            m_builder  = std::make_shared<BuilderT>();
        }
        m_raw = m_builder.get();
    }

    void reserve( int64_t n )
    {
        auto st = m_builder -> Reserve( n );
        if( !st.ok() ) CSP_THROW( RuntimeException, "Failed to reserve builder capacity: " << st.ToString() );
    }

    // Append a set value. Safe variant grows as needed; unsafe variant assumes prior reserve()
    // (fixed-width types only; for variable-length it falls through to the safe path).
    void append( const CspT & v )       { doAppend<false>( v ); }
    void appendUnsafe( const CspT & v ) { doAppend<true>( v ); }

    void appendNull()
    {
        auto st = m_builder -> AppendNull();
        if( !st.ok() ) CSP_THROW( RuntimeException, "Failed to append null: " << st.ToString() );
    }

    std::shared_ptr<::arrow::Array> finish()
    {
        std::shared_ptr<::arrow::Array> arr;
        auto st = m_builder -> Finish( &arr );
        if( !st.ok() ) CSP_THROW( RuntimeException, "Failed to finish array: " << st.ToString() );
        return arr;
    }

    const std::shared_ptr<::arrow::DataType> &   dataType() const { return m_dataType; }
    const std::shared_ptr<::arrow::ArrayBuilder> & builder() const { return m_builder; }

private:
    template<bool Unsafe, class BuilderT, class V>
    void appendFixed( V val )
    {
        auto * b = static_cast<BuilderT *>( m_raw );
        if constexpr( Unsafe )
            b -> UnsafeAppend( val );
        else
        {
            auto st = b -> Append( val );
            if( !st.ok() ) CSP_THROW( RuntimeException, "Failed to append value: " << st.ToString() );
        }
    }

    void appendVarLen( const char * data, int32_t len )
    {
        ::arrow::Status st;
        if constexpr( std::is_same_v<CspT, std::string> )
            st = m_isBytes ? static_cast<::arrow::BinaryBuilder *>( m_raw ) -> Append( data, len )
                           : static_cast<::arrow::StringBuilder *>( m_raw ) -> Append( data, len );
        else  // enum
            st = static_cast<::arrow::StringBuilder *>( m_raw ) -> Append( data, len );
        if( !st.ok() ) CSP_THROW( RuntimeException, "Failed to append string/bytes: " << st.ToString() );
    }

    template<bool Unsafe>
    void doAppend( const CspT & v )
    {
        if constexpr( std::is_same_v<CspT, DateTime> )
            appendFixed<Unsafe, ::arrow::TimestampBuilder>( v.asNanoseconds() );
        else if constexpr( std::is_same_v<CspT, TimeDelta> )
            appendFixed<Unsafe, ::arrow::DurationBuilder>( v.asNanoseconds() );
        else if constexpr( std::is_same_v<CspT, Time> )
            appendFixed<Unsafe, ::arrow::Time64Builder>( v.asNanoseconds() );
        else if constexpr( std::is_same_v<CspT, Date> )
            appendFixed<Unsafe, ::arrow::Date32Builder>(
                static_cast<int32_t>( DateTime( v.year(), v.month(), v.day() ).asNanoseconds() / csp::NANOS_PER_DAY ) );
        else if constexpr( std::is_same_v<CspT, std::string> )
            appendVarLen( v.c_str(), static_cast<int32_t>( v.length() ) );
        else if constexpr( std::is_same_v<CspT, CspEnum> )
        {
            auto & n = v.name();
            appendVarLen( n.c_str(), static_cast<int32_t>( n.length() ) );
        }
        else if constexpr( std::is_same_v<CspT, bool> )
            appendFixed<Unsafe, ::arrow::BooleanBuilder>( v );
        else  // numeric
        {
            using BuilderT = typename ::arrow::CTypeTraits<CspT>::BuilderType;
            appendFixed<Unsafe, BuilderT>( static_cast<typename BuilderT::value_type>( v ) );
        }
    }

    bool                                     m_isBytes;
    std::shared_ptr<::arrow::DataType>       m_dataType;
    std::shared_ptr<::arrow::ArrayBuilder>   m_builder;
    ::arrow::ArrayBuilder *                  m_raw = nullptr;
};

} // namespace csp::adapters::arrow

#endif // _IN_CSP_ADAPTERS_ARROW_ArrowScalarWriter_H
