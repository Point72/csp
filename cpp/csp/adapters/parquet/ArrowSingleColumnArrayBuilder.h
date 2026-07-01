#ifndef _IN_CSP_ADAPTERS_PARQUET_ArrowSingleColumnArrayBuilder_H
#define _IN_CSP_ADAPTERS_PARQUET_ArrowSingleColumnArrayBuilder_H

#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/adapters/arrow/ArrowScalarWriter.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <csp/engine/Struct.h>
#include <arrow/builder.h>
#include <optional>
#include <string>

namespace csp::adapters::parquet
{
class ArrowSingleColumnArrayBuilder
{
public:
    ArrowSingleColumnArrayBuilder( std::string columnName, std::uint32_t chunkSize )
            : m_columnName( std::move( columnName ) ), m_chunkSize( chunkSize )
    {
    }

    ArrowSingleColumnArrayBuilder( const ArrowSingleColumnArrayBuilder &other ) = delete;
    ArrowSingleColumnArrayBuilder &operator=( const ArrowSingleColumnArrayBuilder &other ) = delete;

    virtual ~ArrowSingleColumnArrayBuilder() {}

    virtual std::shared_ptr<::arrow::DataType> getDataType() = 0;

    virtual int64_t length() const = 0;

    const std::string &getColumnName(){ return m_columnName; }

    std::uint32_t getChunkSize() const{ return m_chunkSize; }

    // Called for each row of the output parquet row, if no value was provided a null should be appended
    virtual void handleRowFinished() = 0;
    // Release the array from the currently built values
    virtual std::shared_ptr<::arrow::Array> buildArray() = 0;

    // Pre-reserve builder capacity for numRows rows. Default no-op; fixed-width builders override to
    // reserve their Arrow builder so the per-row append path avoids repeated reallocation.
    virtual void reserve( int64_t /*numRows*/ ) {}

private:
    const std::string   m_columnName;
    const std::uint32_t m_chunkSize;
};

class ListColumnArrayBuilder : public ArrowSingleColumnArrayBuilder
{
public:
    using ColumnBuilderPtr = std::shared_ptr<ArrowSingleColumnArrayBuilder>;

    ListColumnArrayBuilder( std::string columnName, std::uint32_t chunkSize,
                            const std::shared_ptr<::arrow::ArrayBuilder>& valueBuilder,
                            csp::adapters::arrow::ListItemsWriter writeItemsFn )
            : ArrowSingleColumnArrayBuilder( std::move( columnName ), chunkSize ),
              m_valueBuilder( valueBuilder ),
              m_builderPtr( std::make_shared<::arrow::ListBuilder>( ::arrow::default_memory_pool(), m_valueBuilder ) ),
              m_writeItemsFn( std::move( writeItemsFn ) )
    {

    }

    virtual std::shared_ptr<::arrow::DataType> getDataType() override
    {
        return m_builderPtr -> type();
    }

    virtual int64_t length() const override
    {
        return m_builderPtr -> length();
    }

    // Called for each row of the output parquet row, if no value was provided a null should be appended
    virtual void handleRowFinished() override
    {
        if( m_value.has_value() )
        {
            CSP_TRUE_OR_THROW_RUNTIME( m_builderPtr -> Append().ok(), "Failed to append list" );
            m_writeItemsFn( m_value.value() );
            m_value.reset();
        }
        else
        {
            STATUS_OK_OR_THROW_RUNTIME( m_builderPtr -> AppendNull(), "Failed write null arrow list" );
        }
    }

    // Release the array from the currently built values
    virtual std::shared_ptr<::arrow::Array> buildArray() override
    {
        std::shared_ptr<::arrow::Array> array;
        STATUS_OK_OR_THROW_RUNTIME( m_builderPtr -> Finish( &array ), "Failed to create arrow list array" );
        return array;
    }

    void setValue( const DialectGenericType& value)
    {
        m_value = value;
    }

private:
    std::shared_ptr<::arrow::ArrayBuilder>                m_valueBuilder;
    std::shared_ptr<::arrow::ListBuilder>                 m_builderPtr;
    csp::adapters::arrow::ListItemsWriter                 m_writeItemsFn;
    std::optional<DialectGenericType>                      m_value;
};


// Scalar single-column builder: appends the per-row value (or null) directly via the shared
// ArrowScalarColumnWriter<CspT> kernel -- no scratch struct. Mirrors ListColumnArrayBuilder's
// pending-value model: setValue() marks this row's value, handleRowFinished() appends value-or-null.
template<typename CspT>
class ScalarColumnArrayBuilder : public ArrowSingleColumnArrayBuilder
{
public:
    ScalarColumnArrayBuilder( std::string columnName, std::uint32_t chunkSize, bool isBytes = false )
            : ArrowSingleColumnArrayBuilder( std::move( columnName ), chunkSize ),
              m_writer( isBytes )
    {
        m_writer.reserve( chunkSize );
    }

    std::shared_ptr<::arrow::DataType> getDataType() override { return m_writer.dataType(); }

    int64_t length() const override { return m_writer.builder() -> length(); }

    void handleRowFinished() override
    {
        if( m_value.has_value() )
        {
            m_writer.append( m_value.value() );
            m_value.reset();
        }
        else
        {
            m_writer.appendNull();
        }
    }

    std::shared_ptr<::arrow::Array> buildArray() override
    {
        auto array = m_writer.finish();
        // finish() resets builder capacity; re-reserve so the per-row append path stays allocation-free.
        m_writer.reserve( getChunkSize() );
        return array;
    }

    void reserve( int64_t numRows ) override { m_writer.reserve( numRows ); }

    // Mark this row's value (used by the output adapter on tick and by dict-basket writeValue).
    void setValue( const CspT & value ) { m_value = value; }

private:
    csp::adapters::arrow::ArrowScalarColumnWriter<CspT> m_writer;
    std::optional<CspT>                                 m_value;
};


}

#endif
