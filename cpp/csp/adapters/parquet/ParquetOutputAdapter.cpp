#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ParquetWriter.h>

using namespace csp;
using namespace csp::adapters::parquet;

namespace csp::adapters::parquet
{

uint32_t ParquetOutputHandler::getChunkSize() const
{
    return m_parquetWriter.getChunkSize();
}


SingleColumnParquetOutputHandler::SingleColumnParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type,
                                                                    std::string columnName )
        : ParquetOutputHandler( parquetWriter, type )
{
    switch( m_type -> type() )
    {
        case CspType::TypeTraits::BOOL:
            createColumnBuilder<BoolArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::INT8:
            createColumnBuilder<Int8ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::UINT8:
            createColumnBuilder<UInt8ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::INT16:
            createColumnBuilder<Int16ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::UINT16:
            createColumnBuilder<UInt16ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::INT32:
            createColumnBuilder<Int32ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::UINT32:
            createColumnBuilder<UInt32ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::INT64:
            createColumnBuilder<Int64ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::UINT64:
            createColumnBuilder<UInt64ArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::DOUBLE:
            createColumnBuilder<DoubleArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::DATETIME:
            createColumnBuilder<DatetimeArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::TIMEDELTA:
            createColumnBuilder<TimedeltaArrayBuilder>( columnName );
            break;
        case CspType::TypeTraits::DATE:
            createColumnBuilder<DateArrayBuilder>( columnName );
            break;
       case CspType::TypeTraits::TIME:
           createColumnBuilder<TimeArrayBuilder>( columnName );
           break;

        case CspType::TypeTraits::STRING:
        {
            const CspStringType &strType = static_cast<const CspStringType &>(*type);
            if( strType.isBytes() )
            {
                createColumnBuilder<BytesArrayBuilder>( columnName );
            }
            else
            {
                createColumnBuilder<StringArrayBuilder>( columnName );
            }
            break;
        }
        case CspType::TypeTraits::ENUM:
        {
            auto enumMetaPtr = std::static_pointer_cast<const CspEnumType>( type ) -> meta();
            createEnumColumnBuilder( columnName, enumMetaPtr );
            break;
        }
        default:
        {
            CSP_THROW( TypeError, "Writing of " << m_type -> type().asString() << " to parquet is not supported" );
        }
    }
}


template< typename ColumnBuilder >
inline void SingleColumnParquetOutputHandler::createColumnBuilder( const std::string &columnName )
{
    m_columnArrayBuilder = std::make_unique<ColumnBuilder>( columnName, getChunkSize() );
    m_valueHandler       = std::make_unique<ValueHandler>(
            [ this ]( const TimeSeriesProvider *input )
            {
                static_cast<ColumnBuilder *>(this -> m_columnArrayBuilder.get())
                        -> setValue( input -> lastValueTyped<typename ColumnBuilder::ValueTypeT>() );
            } );
}

void SingleColumnParquetOutputHandler::createEnumColumnBuilder( const std::string &columnName, CspEnumMeta::Ptr enumMetaPtr )
{
    m_columnArrayBuilder = std::make_unique<StringArrayBuilder>( columnName, getChunkSize() );
    m_valueHandler       = std::make_unique<ValueHandler>(
            [ this ]( const TimeSeriesProvider *input )
            {
                static_cast<StringArrayBuilder *>(this -> m_columnArrayBuilder.get())
                        -> setValue( input -> lastValueTyped<CspEnum>().name() );
            } );
}

void SingleColumnParquetOutputAdapter::executeImpl()
{
    ( *m_valueHandler )( input() );
    m_parquetWriter.scheduleEndCycleEvent();
}

ListColumnParquetOutputHandler::ListColumnParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &elemType,
                                                                const std::string &columnName,
                                                                DialectGenericListWriterInterface::Ptr &listWriterInterface )
        : ParquetOutputHandler( parquetWriter, CspType::DIALECT_GENERIC() ),
          m_columnArrayBuilder(
                  std::make_shared<ListColumnArrayBuilder>( columnName, getChunkSize(), createValueBuilder( elemType, listWriterInterface ),
                                                            listWriterInterface ) )
{
    m_valueHandler = std::make_unique<ValueHandler>(
            [ this ]( const TimeSeriesProvider *input )
            {
                static_cast<ListColumnArrayBuilder *>(this -> m_columnArrayBuilder.get())
                        -> setValue( input -> lastValueTyped<DialectGenericType>() );
            } );
}

namespace
{
template< typename A, typename V = typename A::value_type >
inline std::shared_ptr<::arrow::ArrayBuilder> makeArrayAndAttachToWriter( DialectGenericListWriterInterface::Ptr &listWriterInterface )
{
    auto&& typedWriter = std::dynamic_pointer_cast<TypedDialectGenericListWriterInterface<V>>( listWriterInterface );
    auto& listWriterInterfaceRef = *listWriterInterface;
    CSP_TRUE_OR_THROW( typedWriter != nullptr, TypeError,
                       "Expected " << typeid( TypedDialectGenericListWriterInterface<V> ).name() << " " << " got " <<
                                   typeid( listWriterInterfaceRef ).name() );

    auto res = std::make_shared<A>();
    typedWriter -> setWriteFunction(
            [ res ]( const V &value ){ STATUS_OK_OR_THROW_RUNTIME( res -> Append( value ), "Failed to append value to list array" ); } );
    return res;
}

}

std::shared_ptr<::arrow::ArrayBuilder> ListColumnParquetOutputHandler::createValueBuilder( const CspTypePtr &elemType,
                                                                                           DialectGenericListWriterInterface::Ptr &listWriterInterface )
{
    switch( elemType -> type() )
    {
        case CspType::TypeTraits::BOOL:
            return makeArrayAndAttachToWriter<arrow::BooleanBuilder>(listWriterInterface);
        case CspType::TypeTraits::INT64:
            return makeArrayAndAttachToWriter<arrow::Int64Builder>(listWriterInterface);
        case CspType::TypeTraits::DOUBLE:
            return makeArrayAndAttachToWriter<arrow::DoubleBuilder>(listWriterInterface);
        case CspType::TypeTraits::STRING:
            return makeArrayAndAttachToWriter<arrow::StringBuilder, std::string>(listWriterInterface);
        default:
        {
            CSP_THROW( TypeError,
                       "Writing of list with elements of type " << elemType -> type().asString() << " to parquet is not supported" );
        }
    }
}

void ListColumnParquetOutputAdapter::executeImpl()
{
    ( *m_valueHandler )( input() );
    m_parquetWriter.scheduleEndCycleEvent();
}

StructParquetOutputHandler::StructParquetOutputHandler( Engine *engine, ParquetWriter &parquetWriter, CspTypePtr &type,
                                                        DictionaryPtr fieldMap )
        : ParquetOutputHandler( parquetWriter, type )
{
    auto structMetaPtr = std::static_pointer_cast<const CspStructType>( type ) -> meta().get();

    for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
    {
        createColumnBuilder( structMetaPtr, it.value<std::string>(), it.key(), nullptr );
    }
}

void StructParquetOutputHandler::writeValueFromTs( const TimeSeriesProvider *input )
{
    const Struct *structData = input -> lastValueTyped<StructPtr>().get();

    for( auto &&valueHandler: m_valueHandlers )
    {
        valueHandler( structData );
    }
    m_parquetWriter.scheduleEndCycleEvent();
}

inline StructParquetOutputHandler::ColumnBuilderResultType StructParquetOutputHandler::createColumnBuilder(
        const StructMeta *structMeta,
        const std::string &columnName, const std::string &structFieldName,
        const std::string *path )
{
    auto fieldPtr = structMeta -> field( structFieldName ).get();
    switch( fieldPtr -> type() -> type() )
    {
        case CspType::TypeTraits::BOOL:
            return createColumnBuilder<BoolArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::INT8:
            return createColumnBuilder<Int8ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::UINT8:
            return createColumnBuilder<UInt8ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::INT16:
            return createColumnBuilder<Int16ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::UINT16:
            return createColumnBuilder<UInt16ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::INT32:
            return createColumnBuilder<Int32ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::UINT32:
            return createColumnBuilder<UInt32ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::INT64:
            return createColumnBuilder<Int64ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::UINT64:
            return createColumnBuilder<UInt64ArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::DOUBLE:
            return createColumnBuilder<DoubleArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::DATETIME:
            return createColumnBuilder<DatetimeArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::TIMEDELTA:
            return createColumnBuilder<TimedeltaArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::DATE:
            return createColumnBuilder<DateArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::TIME:
            return createColumnBuilder<TimeArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::STRING:
            return createColumnBuilder<StringArrayBuilder>( fieldPtr, columnName, path );
        case CspType::TypeTraits::ENUM:
            return createEnumColumnBuilder( fieldPtr, columnName, path );
        case CspType::TypeTraits::STRUCT:
            return createStructColumnBuilder( fieldPtr, columnName, path );
        default:
            CSP_THROW( TypeError, "Writing of column " << columnName << " of type " << fieldPtr -> type() -> type().asString()
                                                       << " to parquet is not supported" );
    }
}

template< typename ColumnBuilder >
inline StructParquetOutputHandler::ColumnBuilderResultType StructParquetOutputHandler::createColumnBuilder(
        const StructField *field,
        const std::string &columnName,
        const std::string *path )
{
    std::shared_ptr<ColumnBuilder> columnBuilderPtr = std::make_shared<ColumnBuilder>( resolveFullColumnName( path, columnName ),
                                                                                       getChunkSize() );

    auto columnBuilderRawPtr = columnBuilderPtr.get();
    using T = typename ColumnBuilder::ValueTypeT;

    ValueHandler res = [ field, columnBuilderRawPtr ]( const Struct *s )
    {
        if( field -> isSet( s ) )
        {
            columnBuilderRawPtr -> setValue( field -> value<T>( s ) );
        }
    };
    // We need to collect on the top level value handlers and array builders. Value handlers and builders of nested structs are stored
    // in the struct field handlers. Path is non null in this case.
    if( !path )
    {
        m_valueHandlers.push_back( res );
        m_columnArrayBuilders.push_back( columnBuilderPtr );
    }
    return { columnBuilderPtr, res };
}

inline StructParquetOutputHandler::ColumnBuilderResultType StructParquetOutputHandler::createEnumColumnBuilder(
        const StructField *field,
        const std::string &columnName,
        const std::string *path )
{
    auto columnBuilderPtr{ std::make_shared<StringArrayBuilder>( resolveFullColumnName( path, columnName ), getChunkSize() ) };
    auto columnBuilderRawPtr{ columnBuilderPtr.get() };
    auto enumMetaPtr = std::static_pointer_cast<const CspEnumType>( field -> type() ) -> meta();

    ValueHandler res = [ field, columnBuilderRawPtr ]( const Struct *s )
    {
        if( field -> isSet( s ) )
        {
            columnBuilderRawPtr -> setValue( field -> value<CspEnum>( s ).name() );
        }
    };
    if( path == nullptr )
    {
        m_valueHandlers.push_back( res );
        m_columnArrayBuilders.push_back( columnBuilderPtr );
    }
    return { columnBuilderPtr, res };
}

inline StructParquetOutputHandler::ColumnBuilderResultType StructParquetOutputHandler::createStructColumnBuilder(
        const StructField *structField,
        const std::string &columnName,
        const std::string *path )
{
    std::vector<std::shared_ptr<::arrow::Field>>            fields;
    std::vector<StructColumnArrayBuilder::ColumnBuilderPtr> childArrayBuilders;
    std::vector<ValueHandler>                               childFieldSetters;

    auto structFieldMetaPtr = std::static_pointer_cast<const CspStructType>( structField -> type() ) -> meta().get();

    for( auto &subField:structFieldMetaPtr -> fields() )
    {
        std::string fieldSubPath = resolveFullColumnName( path, subField -> fieldname() );

        auto childBuilderRes = createColumnBuilder( structFieldMetaPtr,
                                                    subField -> fieldname(),
                                                    subField -> fieldname(),
                                                    &fieldSubPath );
        childArrayBuilders.push_back( childBuilderRes.m_columnBuilder );
        childFieldSetters.push_back( childBuilderRes.m_valueHandler );
        fields.push_back(
                std::make_shared<::arrow::Field>( subField -> fieldname(), childBuilderRes.m_columnBuilder -> getDataType() ) );
    }

    auto subFieldValueSetter = [ childFieldSetters ]( const Struct *s )
    {
        for( auto &childFieldSetter:childFieldSetters )
        {
            childFieldSetter( s );
        }
    };

    auto columnBuilderPtr{ std::make_shared<StructColumnArrayBuilder>(
            resolveFullColumnName( path, columnName ), getChunkSize(),
            std::make_shared<::arrow::StructType>( fields ),
            childArrayBuilders,
            std::move( subFieldValueSetter )
    ) };

    auto columnBuilderRawPtr = columnBuilderPtr.get();

    ValueHandler valueHandler = [ structField, columnBuilderRawPtr ]( const Struct *s )
    {
        if( structField -> isSet( s ) )
        {
            columnBuilderRawPtr -> setValue( structField -> value<StructPtr>( s ).get() );
        }
    };
    if( path == nullptr )
    {
        m_valueHandlers.push_back( valueHandler );
        m_columnArrayBuilders.push_back( columnBuilderPtr );
    }
    return { columnBuilderPtr, valueHandler };
}

void StructParquetOutputAdapter::executeImpl()
{
    writeValueFromTs( input() );
}

}
