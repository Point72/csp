#ifndef _IN_CSP_ADAPTERS_PARQUET_ArrowSingleColumnArrayBuilder_H
#define _IN_CSP_ADAPTERS_PARQUET_ArrowSingleColumnArrayBuilder_H

#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <csp/engine/Struct.h>
#include <arrow/builder.h>
#include <string>
#include <csp/adapters/parquet/DialectGenericListWriterInterface.h>

namespace csp::adapters::parquet
{
class ArrowSingleColumnArrayBuilder
{
public:
    ArrowSingleColumnArrayBuilder( std::string columnName, std::uint32_t chunkSize )
            : m_columnName( columnName ), m_chunkSize( chunkSize )
    {
    }

    ArrowSingleColumnArrayBuilder( const ArrowSingleColumnArrayBuilder &other ) = delete;
    ArrowSingleColumnArrayBuilder &operator=( const ArrowSingleColumnArrayBuilder &other ) = delete;

    virtual ~ArrowSingleColumnArrayBuilder() {}

    virtual std::shared_ptr<arrow::DataType> getDataType() = 0;
    virtual std::shared_ptr<arrow::ArrayBuilder> getBuilder() = 0;

    virtual int64_t length() const = 0;

    const std::string &getColumnName(){ return m_columnName; }

    std::uint32_t getChunkSize() const{ return m_chunkSize; }

    // Called for each row of the output parquet row, if no value was provided a null should be appended
    virtual void handleRowFinished() = 0;
    // Release the array from the currently built values
    virtual std::shared_ptr<arrow::Array> buildArray() = 0;

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
            : ArrowSingleColumnArrayBuilder( columnName, chunkSize ),
              m_childArrayBuilders( childArrayBuilders ),
              m_builderPtr( std::make_shared<::arrow::StructBuilder>( type, arrow::default_memory_pool(),
                                                                      getArrowChildArrayBuilders( childArrayBuilders ) ) ),
              m_fieldValueSetter( fieldValueSetter ),
              m_hasValue( false )
    {
    }

    virtual std::shared_ptr<arrow::DataType> getDataType() override
    {
        return m_builderPtr -> type();
    }

    virtual std::shared_ptr<arrow::ArrayBuilder> getBuilder() override
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
    virtual std::shared_ptr<arrow::Array> buildArray() override
    {
        std::shared_ptr<arrow::Array> array;
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
                            const std::shared_ptr<arrow::ArrayBuilder>& valueBuilder,
                            const DialectGenericListWriterInterface::Ptr& listWriterInterface)
            : ArrowSingleColumnArrayBuilder( columnName, chunkSize ),
              m_valueBuilder( valueBuilder ),
              m_builderPtr( std::make_shared<::arrow::ListBuilder>( arrow::default_memory_pool(), m_valueBuilder ) ),
              m_listWriterInterface(listWriterInterface)
    {

    }

    virtual std::shared_ptr<arrow::DataType> getDataType() override
    {
        return m_builderPtr -> type();
    }

    virtual std::shared_ptr<arrow::ArrayBuilder> getBuilder() override
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
            m_listWriterInterface->writeItems(m_value.value());
            m_value.reset();
        }
        else
        {
            STATUS_OK_OR_THROW_RUNTIME( m_builderPtr -> AppendNull(), "Failed write null arrow list" );
        }
    }

    // Release the array from the currently built values
    virtual std::shared_ptr<arrow::Array> buildArray() override
    {
        std::shared_ptr<arrow::Array> array;
        STATUS_OK_OR_THROW_RUNTIME( m_builderPtr -> Finish( &array ), "Failed to create arrow list array" );
        return array;
    }

    void setValue( const DialectGenericType& value)
    {
        m_value = value;
    }

private:
    static std::shared_ptr<arrow::ArrayBuilder> createValueArrayBuilderForType( const std::shared_ptr<::arrow::DataType> &childType)
    {
        switch(childType->id())
        {
            case arrow::Type::INT64:
                return std::make_shared<arrow::Int64Builder>();
            case arrow::Type::DOUBLE:
                return std::make_shared<arrow::DoubleBuilder>();
            case arrow::Type::BOOL:
                return std::make_shared<arrow::BooleanBuilder>();
            default:
                CSP_THROW(TypeError, "Trying to create arrow list array builder for unsupported element type " << childType->name());
        }
    }

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
    std::shared_ptr<arrow::ArrayBuilder>   m_valueBuilder;
    std::shared_ptr<::arrow::ListBuilder>  m_builderPtr;
    DialectGenericListWriterInterface::Ptr m_listWriterInterface;
    std::optional<DialectGenericType>      m_value;
};


template< typename ValueType, typename ArrowBuilderType >
class BaseTypedArrayBuilder : public ArrowSingleColumnArrayBuilder
{
public:
    using ValueTypeT = ValueType;
    using ArrowBuilderTypeT = ArrowBuilderType;

    BaseTypedArrayBuilder( std::string columnName, std::uint32_t chunkSize )
            : ArrowSingleColumnArrayBuilder( columnName, chunkSize ),
              m_builderPtr( std::make_shared<ArrowBuilderType>() )
    {
        CSP_TRUE_OR_THROW_RUNTIME( m_builderPtr -> Reserve( getChunkSize() ).ok(), "Failed to reserve arrow array size" );
    }

    std::shared_ptr<arrow::ArrayBuilder> getBuilder() override
    {
        return m_builderPtr;
    }

    std::shared_ptr<arrow::DataType> getDataType() override
    {
        return m_builderPtr -> type();
    }

    virtual int64_t length() const override
    {
        return m_builderPtr -> length();
    }

    void handleRowFinished() override
    {
        if( m_value )
        {
            pushValueToArray();
        }
        else
        {
            STATUS_OK_OR_THROW_RUNTIME( m_builderPtr -> AppendNull(), "Failed to append null to arrow array" );
        }
        m_value = nullptr;
    }

    std::shared_ptr<arrow::Array> buildArray() override
    {
        std::shared_ptr<arrow::Array> array;
        CSP_TRUE_OR_THROW_RUNTIME( m_builderPtr -> Finish( &array ).ok(), "Failed to create arrow array" );
        return array;
    }

    void setValue( const ValueType &value )
    {
        m_value = &value;
    }

protected:
    template< typename ...BuilderArgs >
    BaseTypedArrayBuilder( std::string columnName, std::uint32_t chunkSize, BuilderArgs ...args )
            : ArrowSingleColumnArrayBuilder( columnName, chunkSize ),
              m_builderPtr( std::make_shared<ArrowBuilderType>( args... ) )
    {

    }

    virtual void pushValueToArray() = 0;

protected:
    std::shared_ptr<ArrowBuilderType> m_builderPtr;
    const ValueType                   *m_value = nullptr;
};

template< typename ValueType, typename ArrowBuilderType >
class PrimitiveTypedArrayBuilder : public BaseTypedArrayBuilder<ValueType, ArrowBuilderType>
{
public:
    using BaseTypedArrayBuilder<ValueType, ArrowBuilderType>::BaseTypedArrayBuilder;

protected:
    void pushValueToArray()
    {
        [[maybe_unused]] auto status = this -> m_builderPtr -> Append( *this -> m_value );        
    }
};

class StringArrayBuilder : public BaseTypedArrayBuilder<std::string, arrow::StringBuilder>
{
public:
    using BaseTypedArrayBuilder<std::string, arrow::StringBuilder>::BaseTypedArrayBuilder;

protected:
    void pushValueToArray()
    {
        const std::string &value = *m_value;
        STATUS_OK_OR_THROW_RUNTIME( this -> m_builderPtr -> Append( value.c_str(), value.length() ),
                                    "Failed to append value to arrow array" );
    }
};

class BytesArrayBuilder : public BaseTypedArrayBuilder<std::string, arrow::BinaryBuilder>
{
public:
    using BaseTypedArrayBuilder<std::string, arrow::BinaryBuilder>::BaseTypedArrayBuilder;

protected:
    void pushValueToArray()
    {
        const std::string &value = *m_value;
        STATUS_OK_OR_THROW_RUNTIME( this -> m_builderPtr -> Append( value.c_str(), value.length() ),
                                    "Failed to append value to arrow array" );
    }
};

class [[maybe_unused]] CStringArrayBuilder : public BaseTypedArrayBuilder<const char *, arrow::StringBuilder>
{
public:
    using BaseTypedArrayBuilder<const char *, arrow::StringBuilder>::BaseTypedArrayBuilder;

    [[maybe_unused]] void setValueCopyPtr( const char *value )
    {
        m_ptrCopy = value;
        setValue( m_ptrCopy );
    }

protected:
    void pushValueToArray()
    {
        const char *value = *m_value;
        STATUS_OK_OR_THROW_RUNTIME( this -> m_builderPtr -> Append( value ), "Failed to append value to arrow array" );
    }

private:
    const char *m_ptrCopy;
};


class DatetimeArrayBuilder : public BaseTypedArrayBuilder<csp::DateTime, arrow::TimestampBuilder>
{
public:
    DatetimeArrayBuilder( std::string columnName, std::uint32_t chunkSize )
            : BaseTypedArrayBuilder<csp::DateTime, arrow::TimestampBuilder>( columnName,
                                                                             chunkSize,
                                                                             std::make_shared<arrow::TimestampType>( arrow::TimeUnit::NANO,
                                                                                                                     "UTC" ),
                                                                             arrow::default_memory_pool() )
    {
    }

protected:
    void pushValueToArray()
    {
        STATUS_OK_OR_THROW_RUNTIME( this -> m_builderPtr -> Append( m_value -> asNanoseconds() ),
                                    "Failed to append timestamp value to arrow array" );
    }
};

class DateArrayBuilder : public BaseTypedArrayBuilder<csp::Date, arrow::Date32Builder>
{
public:
    DateArrayBuilder( std::string columnName, std::uint32_t chunkSize )
            : BaseTypedArrayBuilder<csp::Date, arrow::Date32Builder>( columnName,
                                                                      chunkSize )
    {
    }

protected:
    void pushValueToArray()
    {
        auto timedelta = *m_value - getOrigin();
        STATUS_OK_OR_THROW_RUNTIME( this -> m_builderPtr -> Append( timedelta.days() ), "Failed to append date value to arrow array" );
    }

private:
    static csp::Date &getOrigin()
    {
        static csp::Date origin{ 1970, 1, 1 };
        return origin;
    }
};

class TimeArrayBuilder : public BaseTypedArrayBuilder<csp::Time, arrow::Time64Builder>
{
public:
    TimeArrayBuilder( std::string columnName, std::uint32_t chunkSize )
        : BaseTypedArrayBuilder<csp::Time, arrow::Time64Builder>( columnName, chunkSize,
                                                                  std::make_shared<arrow::Time64Type>( arrow::TimeUnit::NANO ), arrow::default_memory_pool() )
    {
    }

protected:
    void pushValueToArray()
    {
        STATUS_OK_OR_THROW_RUNTIME( this -> m_builderPtr -> Append( m_value -> asNanoseconds() ), "Failed to append time value to arrow array" );
    }
};

class TimedeltaArrayBuilder : public BaseTypedArrayBuilder<csp::TimeDelta, arrow::DurationBuilder>
{
public:
    TimedeltaArrayBuilder( std::string columnName, std::uint32_t chunkSize )
            : BaseTypedArrayBuilder<csp::TimeDelta, arrow::DurationBuilder>(
            columnName, chunkSize, std::make_shared<arrow::DurationType>( arrow::TimeUnit::NANO ),
            arrow::default_memory_pool() )
    {
    }

protected:
    void pushValueToArray()
    {
        STATUS_OK_OR_THROW_RUNTIME( this -> m_builderPtr -> Append( m_value -> asNanoseconds() ),
                                    "Failed to append date value to arrow array" );
    }
};

using BoolArrayBuilder = PrimitiveTypedArrayBuilder<bool, arrow::BooleanBuilder>;
using Int8ArrayBuilder = PrimitiveTypedArrayBuilder<std::int8_t, arrow::Int8Builder>;
using Int16ArrayBuilder = PrimitiveTypedArrayBuilder<std::int16_t, arrow::Int16Builder>;
using Int32ArrayBuilder = PrimitiveTypedArrayBuilder<std::int32_t, arrow::Int32Builder>;
using Int64ArrayBuilder = PrimitiveTypedArrayBuilder<std::int64_t, arrow::Int64Builder>;
using UInt8ArrayBuilder = PrimitiveTypedArrayBuilder<std::uint8_t, arrow::UInt8Builder>;
using UInt16ArrayBuilder = PrimitiveTypedArrayBuilder<std::uint16_t, arrow::UInt16Builder>;
using UInt32ArrayBuilder = PrimitiveTypedArrayBuilder<std::uint32_t, arrow::UInt32Builder>;
using UInt64ArrayBuilder = PrimitiveTypedArrayBuilder<std::uint64_t, arrow::UInt64Builder>;
using DoubleArrayBuilder = PrimitiveTypedArrayBuilder<double, arrow::DoubleBuilder>;
}

#endif
