// Per-column readers that extract values from Arrow arrays.
//
// FieldReader provides bindColumn()/readNext() for sequential row processing
// and readValueAt() for random-access without structs.

#ifndef _IN_CSP_ADAPTERS_ARROW_ArrowFieldReader_H
#define _IN_CSP_ADAPTERS_ARROW_ArrowFieldReader_H

#include <csp/engine/Struct.h>
#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <arrow/array.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace csp::adapters::arrow
{

inline int64_t timeUnitMultiplier( ::arrow::TimeUnit::type unit )
{
    switch( unit )
    {
        case ::arrow::TimeUnit::SECOND: return csp::NANOS_PER_SECOND;
        case ::arrow::TimeUnit::MILLI:  return csp::NANOS_PER_MILLISECOND;
        case ::arrow::TimeUnit::MICRO:  return csp::NANOS_PER_MICROSECOND;
        case ::arrow::TimeUnit::NANO:   return 1LL;
        default:
            CSP_THROW( TypeError, "Unexpected arrow TimeUnit: " << static_cast<int>( unit ) );
    }
}

class FieldReader
{
public:
    virtual ~FieldReader() = default;

    FieldReader( const std::string & columnName, const StructFieldPtr & field )
        : m_field( field ), m_columnNames( { columnName } )
    {
    }

    FieldReader( std::vector<std::string> columnNames, const StructFieldPtr & field )
        : m_field( field ), m_columnNames( std::move( columnNames ) )
    {
    }

    // Bind to a new column array, resetting the row counter.
    void bindColumn( const ::arrow::Array * column )
    {
        m_column = column;
        m_row = 0;
        onBind();
    }

    // Batch-level bind for multi-column readers (e.g. numpy NDArray).
    virtual void bindBatch( const ::arrow::RecordBatch & batch ) {}

    // Read current row into struct and advance.
    void readNext( Struct * s )
    {
        doReadNext( m_row, s );
        ++m_row;
    }

    // Read value at a specific row into optional<T> (as void*) without advancing.
    void readValueAt( int64_t row, void * optionalOut )
    {
        doReadNextValue( row, optionalOut );
    }

    // Advance without reading (keeps child readers in sync for null parent rows).
    virtual void skipNext()
    {
        ++m_row;
    }

    // Columnar bulk-read into pre-allocated structs.
    virtual void readAll( std::vector<StructPtr> & structs, int64_t numRows )
    {
        for( int64_t row = 0; row < numRows; ++row )
            doReadNext( row, structs[row].get() );
        m_row = numRows;
    }

    const std::vector<std::string> & columnNames() const { return m_columnNames; }
    const StructFieldPtr & field() const { return m_field; }

protected:
    // Called after bindColumn; subclasses cache derived pointers (e.g. dictionaries).
    virtual void onBind() {}

    virtual void doReadNext( int64_t row, Struct * s ) = 0;
    virtual void doReadNextValue( int64_t row, void * optionalOut ) = 0;

    StructFieldPtr             m_field;
    std::vector<std::string>   m_columnNames;
    const ::arrow::Array *     m_column = nullptr;
    int64_t                    m_row    = 0;
};

// Typed intermediate: doExtract(row, ValueT&) → bool provides doReadNext/doReadNextValue.
template<typename ValueT>
class TypedFieldReader : public FieldReader
{
public:
    using FieldReader::FieldReader;  // inherit constructors

protected:
    virtual bool doExtract( int64_t row, ValueT & out ) = 0;

    void doReadNext( int64_t row, Struct * s ) override
    {
        ValueT tmp{};
        if( doExtract( row, tmp ) )
            m_field -> template setValue<ValueT>( s, std::move( tmp ) );
    }

    void doReadNextValue( int64_t row, void * optionalOut ) override
    {
        auto & out = *static_cast<std::optional<ValueT> *>( optionalOut );
        ValueT tmp{};
        if( doExtract( row, tmp ) )
            out = std::move( tmp );
        else
            out.reset();
    }
};

// Columnar bulk-read helper: dispatches fn(arr, row, struct*) for each row,
// skipping nulls when null_count > 0.
template<typename ArrowArrayT, typename Fn>
inline void readColumn( const ArrowArrayT & typed, std::vector<StructPtr> & structs, int64_t numRows, Fn && fn )
{
    if( typed.null_count() == 0 )
        for( int64_t i = 0; i < numRows; ++i )
            fn( typed, i, structs[i].get() );
    else
        for( int64_t i = 0; i < numRows; ++i )
            if( typed.IsValid( i ) )
                fn( typed, i, structs[i].get() );
}

// Concrete FieldReader that extracts values via a stateless or capturing lambda.
// This is the single point of array access for scalar columns.
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

    // Non-virtual direct read: called from TypedColumnDispatcher to avoid
    // virtual dispatch through FieldReader::doReadNextValue.
    void readDirect( int64_t row, std::optional<ValueT> & out )
    {
        auto & typed = static_cast<const ArrowArrayT &>( *this -> m_column );
        if( typed.IsValid( row ) )
            out = m_extractFn( typed, row );
        else
            out.reset();
    }

    const ExtractFn & extractFn() const { return m_extractFn; }

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

    void doReadNextValue( int64_t row, void * optionalOut ) override
    {
        auto & out = *static_cast<std::optional<ValueT> *>( optionalOut );
        auto & typed = static_cast<const ArrowArrayT &>( *this -> m_column );
        if( typed.IsValid( row ) )
            out = m_extractFn( typed, row );
        else
            out.reset();
    }

private:
    ExtractFn  m_extractFn;
};

// Factory: create a FieldReader for a given Arrow field + CSP struct field.
// Optional structMeta for STRUCT columns when structField is nullptr.
std::unique_ptr<FieldReader> createFieldReader(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const StructFieldPtr & structField,
    const std::shared_ptr<const StructMeta> & structMeta = nullptr
);

// Factory for list/array column readers (registered by Python-aware layer).
using ListFieldReaderFactory = std::function<
    std::unique_ptr<FieldReader>( const std::shared_ptr<::arrow::Field> &, const StructFieldPtr & )>;

void registerListFieldReaderFactory( ListFieldReaderFactory factory );

}

#endif // _IN_CSP_ADAPTERS_ARROW_ArrowFieldReader_H
