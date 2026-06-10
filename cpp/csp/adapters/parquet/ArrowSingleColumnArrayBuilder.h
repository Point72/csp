#ifndef _IN_CSP_ADAPTERS_PARQUET_ArrowSingleColumnArrayBuilder_H
#define _IN_CSP_ADAPTERS_PARQUET_ArrowSingleColumnArrayBuilder_H

#include <csp/adapters/arrow/ArrowFieldWriter.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <csp/engine/Struct.h>
#include <arrow/builder.h>
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
    virtual std::shared_ptr<::arrow::ArrayBuilder> getBuilder() = 0;

    virtual int64_t length() const = 0;

    const std::string &getColumnName(){ return m_columnName; }

    std::uint32_t getChunkSize() const{ return m_chunkSize; }

    // Called for each row of the output parquet row, if no value was provided a null should be appended
    virtual void handleRowFinished() = 0;
    // Release the array from the currently built values
    virtual std::shared_ptr<::arrow::Array> buildArray() = 0;

private:
    const std::string   m_columnName;
    const std::uint32_t m_chunkSize;
};

class StructColumnArrayBuilder : public ArrowSingleColumnArrayBuilder
{
public:
    using ColumnBuilderPtr = std::shared_ptr<ArrowSingleColumnArrayBuilder>;
    using FieldValueSetter = std::function<void( const Struct * )>;

    StructColumnArrayBuilder( std::string columnName, std::uint32_t chunkSize,
                              const std::shared_ptr<::arrow::DataType> &type,
                              const std::vector<ColumnBuilderPtr> &childArrayBuilders,
                              FieldValueSetter fieldValueSetter )
            : ArrowSingleColumnArrayBuilder( std::move( columnName ), chunkSize ),
              m_childArrayBuilders( childArrayBuilders ),
              m_builderPtr( std::make_shared<::arrow::StructBuilder>( type, ::arrow::default_memory_pool(),
                                                                      getArrowChildArrayBuilders( childArrayBuilders ) ) ),
              m_fieldValueSetter( fieldValueSetter ),
              m_hasValue( false )
    {
    }

    virtual std::shared_ptr<::arrow::DataType> getDataType() override
    {
        return m_builderPtr -> type();
    }

    virtual std::shared_ptr<::arrow::ArrayBuilder> getBuilder() override
    {
        return m_builderPtr;
    }

    virtual int64_t length() const override
    {
        return m_builderPtr -> length();
    }

    // Called for each row of the output parquet row, if no value was provided a null should be appended
    virtual void handleRowFinished() override
    {
        if( m_hasValue )
        {
            m_hasValue = false;
            for( auto &childBuilder : m_childArrayBuilders )
            {
                childBuilder -> handleRowFinished();
            }
            CSP_TRUE_OR_THROW_RUNTIME( m_builderPtr -> Append().ok(), "Failed to append struct" );
        }
        else
        {
            STATUS_OK_OR_THROW_RUNTIME( m_builderPtr -> AppendNull(), "Failed to create arrow array" );
        }
    }

    // Release the array from the currently built values
    virtual std::shared_ptr<::arrow::Array> buildArray() override
    {
        std::shared_ptr<::arrow::Array> array;
        STATUS_OK_OR_THROW_RUNTIME( m_builderPtr -> Finish( &array ), "Failed to create arrow array" );
        return array;
    }

    void setValue( const Struct *value )
    {
        m_hasValue = true;
        m_fieldValueSetter( value );
    }

private:
    static std::vector<std::shared_ptr<::arrow::ArrayBuilder>> getArrowChildArrayBuilders(
            const std::vector<ColumnBuilderPtr> &childArrayBuilders )
    {
        std::vector<std::shared_ptr<::arrow::ArrayBuilder>> res;

        for( auto &columnBuilder : childArrayBuilders )
        {
            res.push_back( columnBuilder -> getBuilder() );
        }
        return res;
    }

private:
    std::vector<ColumnBuilderPtr>           m_childArrayBuilders;
    std::shared_ptr<::arrow::StructBuilder> m_builderPtr;
    FieldValueSetter                        m_fieldValueSetter;
    bool                                    m_hasValue;
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

    virtual std::shared_ptr<::arrow::ArrayBuilder> getBuilder() override
    {
        return m_builderPtr;
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


}

#endif
