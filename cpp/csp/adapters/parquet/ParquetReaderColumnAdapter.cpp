#include <csp/adapters/parquet/ParquetReaderColumnAdapter.h>
#include <csp/adapters/parquet/ParquetReader.h>
#include <csp/engine/TypeCast.h>

#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/array.h>

namespace csp::adapters::parquet
{

template< typename ArrowArrayType >
static inline std::unique_ptr<ParquetColumnAdapter> createDateColumnAdapter(
        ParquetReader &reader,
        const std::string& columnName,
        const std::shared_ptr<typename ArrowArrayType::TypeClass> &dateType )
{
    switch( dateType -> unit() )
    {
        case arrow::DateUnit::MILLI:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new DateColumnAdapter<1000000, ArrowArrayType>( reader, columnName ) );
        case arrow::DateUnit::DAY:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new DateColumnAdapter<1000000000LL * 3600 * 24, ArrowArrayType>( reader, columnName ) );
    }
    CSP_THROW( csp::TypeError, "Unexpected day unit: " << ( int ) dateType -> unit() << " for column " << columnName );
}

template< typename ArrowArrayType >
static inline std::unique_ptr<ParquetColumnAdapter> createTimeColumnAdapter(
        ParquetReader &reader,
        const std::string& columnName,
        const std::shared_ptr<typename ArrowArrayType::TypeClass> &timeType )
{
    ParquetColumnAdapter * adapter;
    switch( timeType -> unit() )
    {
        case arrow::TimeUnit::SECOND: adapter = new TimeColumnAdapter<1000000000,ArrowArrayType>( reader, columnName ); break;
        case arrow::TimeUnit::MILLI:  adapter = new TimeColumnAdapter<1000000,ArrowArrayType>( reader, columnName ); break;
        case arrow::TimeUnit::MICRO:  adapter = new TimeColumnAdapter<1000,ArrowArrayType>( reader, columnName ); break;
        case arrow::TimeUnit::NANO:   adapter = new TimeColumnAdapter<1,ArrowArrayType>( reader, columnName ); break;
    }

    return std::unique_ptr<ParquetColumnAdapter>( adapter );
}

template< typename ArrowListArrayType >
static inline std::unique_ptr<ParquetColumnAdapter> createListColumnAdapter(
        ParquetReader &parquetReader,
        const ::arrow::Field &field,
        const std::string &fileName )
{
    auto listType  = std::static_pointer_cast<arrow::ListType>( field.type() );
    auto valueType = listType -> value_type();
    switch( valueType -> id() )
    {
        case arrow::Type::INT64:
            return std::make_unique<NativeListColumnAdapter<ArrowListArrayType, arrow::Int64Array>>( parquetReader, field.name() );
        case arrow::Type::DOUBLE:
            return std::make_unique<NativeListColumnAdapter<ArrowListArrayType, arrow::DoubleArray>>( parquetReader, field.name() );
        case arrow::Type::BOOL:
            return std::make_unique<NativeListColumnAdapter<ArrowListArrayType, arrow::BooleanArray>>( parquetReader, field.name() );
        case arrow::Type::STRING:
            return std::make_unique<BytesListColumnAdapter<ArrowListArrayType, arrow::StringArray>>( parquetReader, field.name() );
        case arrow::Type::BINARY:
            return std::make_unique<BytesListColumnAdapter<ArrowListArrayType, arrow::BinaryArray>>( parquetReader, field.name() );
        case arrow::Type::LARGE_STRING:
            return std::make_unique<BytesListColumnAdapter<ArrowListArrayType, arrow::LargeStringArray>>( parquetReader, field.name() );
        case arrow::Type::LARGE_BINARY:
            return std::make_unique<BytesListColumnAdapter<ArrowListArrayType, arrow::LargeBinaryArray>>( parquetReader, field.name() );
        default:
            CSP_THROW( TypeError,
                       "Trying to create arrow list array reader for unsupported element type " << valueType -> name()
                       << " for column " << field.name() << " in file " << fileName );
    }
}

std::unique_ptr<ParquetColumnAdapter> createColumnAdapter(
        ParquetReader &parquetReader,
        const ::arrow::Field &field,
        const std::string &fileName,
        const std::map<std::string, std::shared_ptr<StructMeta>>* structMetaByColumnName )
{
    auto typeId = field.type() -> id();
    switch( typeId )
    {
        case arrow::Type::BOOL:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<bool, arrow::BooleanArray>( parquetReader, field.name() ) );
        case arrow::Type::UINT8:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::uint8_t, arrow::UInt8Array>( parquetReader, field.name() ) );
        case arrow::Type::INT8:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::int8_t, arrow::Int8Array>( parquetReader, field.name() ) );
        case arrow::Type::UINT16:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::uint16_t, arrow::UInt16Array>( parquetReader, field.name() ) );
        case arrow::Type::INT16:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::int16_t, arrow::Int16Array>( parquetReader, field.name() ) );
        case arrow::Type::UINT32:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::uint32_t, arrow::UInt32Array>( parquetReader, field.name() ) );
        case arrow::Type::INT32:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::int32_t, arrow::Int32Array>( parquetReader, field.name() ) );
        case arrow::Type::UINT64:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::uint64_t, arrow::UInt64Array>( parquetReader, field.name() ) );
        case arrow::Type::INT64:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<std::int64_t, arrow::Int64Array>( parquetReader, field.name() ) );
        case arrow::Type::HALF_FLOAT:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<double, arrow::HalfFloatArray>( parquetReader, field.name() ) );
        case arrow::Type::FLOAT:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<double, arrow::FloatArray>( parquetReader, field.name() ) );
        case arrow::Type::DOUBLE:
            return std::unique_ptr<ParquetColumnAdapter>(
                    new NativeTypeColumnAdapter<double, arrow::DoubleArray>( parquetReader, field.name() ) );
        case arrow::Type::STRING:
            return std::unique_ptr<ParquetColumnAdapter>( new StringColumnAdapter<arrow::StringArray>( parquetReader, field.name() ) );
        case arrow::Type::BINARY:
            return std::unique_ptr<ParquetColumnAdapter>( new BytesColumnAdapter<arrow::BinaryArray>( parquetReader, field.name() ) );
        case arrow::Type::DATE32:
            return createDateColumnAdapter<arrow::Date32Array>( parquetReader,
                                                                field.name(),
                                                                std::static_pointer_cast<arrow::Date32Type>(
                                                                        field.type() ) );
        case arrow::Type::DATE64:
            return createDateColumnAdapter<arrow::Date64Array>( parquetReader,
                                                                field.name(),
                                                                std::static_pointer_cast<arrow::Date64Type>(
                                                                        field.type() ) );

        case arrow::Type::TIME32:
            return createTimeColumnAdapter<arrow::Time32Array>( parquetReader, field.name(),
                                                                std::static_pointer_cast<arrow::Time32Type>( field.type() ) );

        case arrow::Type::TIME64:
            return createTimeColumnAdapter<arrow::Time64Array>( parquetReader, field.name(),
                                                                std::static_pointer_cast<arrow::Time64Type>( field.type() ) );

        case arrow::Type::TIMESTAMP:
        {
            auto timestampType = std::static_pointer_cast<arrow::TimestampType>( field.type() );
//            CSP_TRUE_OR_THROW_RUNTIME(
//                    timestampType -> timezone().empty() || timestampType -> timezone() == "UTC" || timestampType -> timezone() == "utc",
//                    "Unexpected parquet column timezone: " << timestampType -> timezone() );
            switch( timestampType -> unit() )
            {
                case arrow::TimeUnit::SECOND:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DatetimeColumnAdapter<1000000000>( parquetReader, field.name() ) );
                case arrow::TimeUnit::MILLI:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DatetimeColumnAdapter<1000000>( parquetReader, field.name() ) );
                case arrow::TimeUnit::MICRO:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DatetimeColumnAdapter<1000>( parquetReader, field.name() ) );
                case arrow::TimeUnit::NANO:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DatetimeColumnAdapter<1>( parquetReader, field.name() ) );
            }
        }
        case arrow::Type::DURATION:
        {
            auto durationType = std::static_pointer_cast<arrow::DurationType>( field.type() );
            switch( durationType -> unit() )
            {
                case arrow::TimeUnit::SECOND:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DurationColumnAdapter<1000000000>( parquetReader, field.name() ) );
                case arrow::TimeUnit::MILLI:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DurationColumnAdapter<1000000>( parquetReader, field.name() ) );
                case arrow::TimeUnit::MICRO:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DurationColumnAdapter<1000>( parquetReader, field.name() ) );
                case arrow::TimeUnit::NANO:
                    return std::unique_ptr<ParquetColumnAdapter>(
                            new DurationColumnAdapter<1>( parquetReader, field.name() ) );
            }
        }
        case arrow::Type::FIXED_SIZE_BINARY:
        {
            return std::make_unique<FixedSizeBinaryColumnAdapter>( parquetReader, field.name() );
        }
        case arrow::Type::DICTIONARY:
        {
            auto dictionaryType = std::static_pointer_cast<arrow::DictionaryType>( field.type() );

            if( dictionaryType -> value_type() -> id() != arrow::Type::STRING )
            {
                CSP_THROW( ParquetColumnTypeError,
                           "Unsupported dictionary column type " << field.type() -> name() << " in file " << fileName
                                                                 << " only string values supported" );
            }
            return std::make_unique<DictionaryColumnAdapter>( parquetReader, field.name() );
        }
        case arrow::Type::STRUCT:
        {
            auto res = std::make_unique<StructColumnAdapter>( parquetReader, std::static_pointer_cast<::arrow::StructType>( field.type() ),
                                                              field.name() );
            if(structMetaByColumnName!= nullptr)
            {
                auto metaIt = structMetaByColumnName->find( field.name() );
                if(metaIt != structMetaByColumnName->end())
                {
                    res->setStructMeta(metaIt->second);
                }
            }
            return res;
        }
        case arrow::Type::LIST:
            return createListColumnAdapter<arrow::ListArray>( parquetReader, field, fileName );
        case arrow::Type::LARGE_STRING:
            return std::unique_ptr<ParquetColumnAdapter>( new StringColumnAdapter<arrow::LargeStringArray>( parquetReader, field.name() ) );
        case arrow::Type::LARGE_BINARY:
            return std::unique_ptr<ParquetColumnAdapter>( new BytesColumnAdapter<arrow::LargeBinaryArray>( parquetReader, field.name() ) );
        case arrow::Type::LARGE_LIST:
            return createListColumnAdapter<arrow::LargeListArray>( parquetReader, field, fileName );
        default:
            CSP_THROW( ParquetColumnTypeError,
                       "Unsupported column type " << field.type() -> name() << " for column " << field.name() << " in file " << fileName );
    }
}

std::unique_ptr<ParquetColumnAdapter> createMissingColumnAdapter(ParquetReader &parquetReader,const std::string& columnName)
{
    return std::make_unique<MissingColumnAdapter>( parquetReader, columnName );
}

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT >
void BaseTypedColumnAdapter<ValueType, ArrowArrayType, ValueDispatcherT>::dispatchValue( const utils::Symbol *symbol )
{
    if( m_curValue.has_value() )
    {
        m_dispatcher.dispatch( &m_curValue.value(), symbol );
    }
    else
    {
        m_dispatcher.dispatch( nullptr, symbol );
    }
}

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT >
void BaseTypedColumnAdapter<ValueType, ArrowArrayType, ValueDispatcherT>::ensureType( CspType::Ptr cspType )
{
    try
    {
        CompatibleTypeSwitch::invoke( cspType.get(), [ cspType, this ]( auto tag )
        {
            // No need to check native types here, they won't be assignable so it will raise exception.
            if( cspType -> type() > CspType::Type::MAX_NATIVE_TYPE )
            {
                CSP_TRUE_OR_THROW( ( std::is_same<ValueType, typename decltype(tag)::type>::value ), TypeError,
                                   "Unexpected column type for column " << getColumnName() << " , expected " << cspType -> type().asCString() <<
                                                                       " got " << ArrowArrayType::TypeClass::type_name() );
            }
        } );
    }
    catch( UnsupportedSwitchType &e )
    {
        CSP_THROW( TypeError, "Unexpected column type for column " << getColumnName() << " , expected " <<  cspType -> type().asCString() <<
                                                                   " got " << ArrowArrayType::TypeClass::type_name() );
    }
}

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT >
void *BaseTypedColumnAdapter<ValueType, ArrowArrayType, ValueDispatcherT>::getCurValueUntyped()
{
    return &m_curValue;
}

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT >
void
BaseTypedColumnAdapter<ValueType, ArrowArrayType, ValueDispatcherT>::handleNewBatch( const std::shared_ptr<::arrow::ChunkedArray> &data )
{
    CSP_TRUE_OR_THROW_RUNTIME( data -> num_chunks() == 1,
                               "Unexpected number of chunks in column" << data -> num_chunks() );
    m_curChunkArray = std::static_pointer_cast<ArrowArrayType>( data -> chunk( 0 ) );
}

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT >
void
BaseTypedColumnAdapter<ValueType, ArrowArrayType, ValueDispatcherT>::handleNewBatch( const std::shared_ptr<::arrow::Array> &data )
{
    m_curChunkArray = std::static_pointer_cast<ArrowArrayType>( data );
}

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT >
void BaseTypedColumnAdapter<ValueType, ArrowArrayType, ValueDispatcherT>::addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol )
{
    try
    {
        auto callback = CompatibleTypeSwitch::invoke( inputAdapter -> type(), [ inputAdapter ]( auto tag )
        {
            return std::function<void(
                    const ValueType * )>( [ inputAdapter ]( const ValueType *val )
                                        {
                                            if( val )
                                            {
                                                inputAdapter -> pushTick<typename decltype(tag)::type>( *val );
                                            }
                                            else
                                            {
                                                inputAdapter -> pushNullTick<typename decltype(tag)::type>();
                                            }
                                        } );

        } );
        m_dispatcher.addSubscriber( callback, symbol );
    }
    catch( UnsupportedSwitchType &e )
    {
        CSP_THROW( TypeError, "Unexpected column type for column " << getColumnName() << " , expected "
                                                                << inputAdapter -> type() -> type().asCString() <<
                                                                " got " << ArrowArrayType::TypeClass::type_name() );
    }
}

template< typename ValueType, typename ArrowArrayType, typename ValueDispatcherT >
void BaseTypedColumnAdapter<ValueType, ArrowArrayType, ValueDispatcherT>::addSubscriber( ManagedSimInputAdapter *inputAdapter,
                    std::optional<utils::Symbol> symbol, const DialectGenericListReaderInterface::Ptr &listReader )
{
    CSP_THROW( NotImplemented, "Trying to subscribe to non-list column using the listReader API?" );
}

ParquetStructAdapter::ParquetStructAdapter( ParquetReader &parquetReader, StructAdapterInfo adapterInfo )
        : m_parquetReader( parquetReader ),
          m_structMeta( std::static_pointer_cast<const CspStructType>( adapterInfo.type() ) -> meta() )
{
    m_resetFunc = [ this, adapterInfo ]()
    {
        m_fieldSetters.clear();
        for( auto it = adapterInfo.fieldMap() -> begin(); it != adapterInfo.fieldMap() -> end(); ++it )
        {
            createFieldSetter( it.value<std::string>(), *m_parquetReader[ it.key() ] );
        }
    };
    m_resetFunc();

    // Note the columns for single struct can come from multiple different readers, so we have to iterate over all columns.
    // MultipleFileParquetReader uses the columns from child SingleFileParquetReader readers.
    for( auto it = adapterInfo.fieldMap() -> begin(); it != adapterInfo.fieldMap() -> end(); ++it )
    {
        auto&& parquetColumnName = it.key();
        auto&& structFieldName = it.value<std::string>();
        auto& columnReader = m_parquetReader[ parquetColumnName ]->getReader();
        columnReader.addDependentStructAdapter(this);
        auto &&fieldPtr = m_structMeta -> field( structFieldName );
        if(fieldPtr -> type() -> type() == CspType::TypeTraits::STRUCT)
        {
            auto &fieldMeta = std::static_pointer_cast<const CspStructType>( fieldPtr -> type() ) -> meta();
            columnReader.setStructColumnMeta(parquetColumnName, fieldMeta);
        }
    }
}

ParquetStructAdapter::ParquetStructAdapter( ParquetReader &parquetReader,
                                            std::shared_ptr<::arrow::StructType> arrowType,
                                            const std::shared_ptr<StructMeta> &structMeta,
                                            const std::vector<std::unique_ptr<ParquetColumnAdapter>> &columnAdapters )
        : m_parquetReader( parquetReader ),
          m_structMeta( structMeta )
{
    m_resetFunc = []()
    {
        CSP_THROW( RuntimeException, "Internal error, trying to reset single column struct adapter" );
    };

    auto &arrowColumns = arrowType -> fields();

    CSP_TRUE_OR_THROW_RUNTIME( arrowColumns.size() == columnAdapters.size(), "Found mismatch between arrow and csp schema" );

    for( std::size_t i = 0; i < arrowColumns.size(); ++i )
    {
        auto &arrowCol = *arrowColumns[ i ];
        createFieldSetter( arrowCol.name(), *columnAdapters[ i ] );
    }
}

void ParquetStructAdapter::createFieldSetter( const std::string &fieldName,
                                              ParquetColumnAdapter &columnAdapter )
{
    if(columnAdapter.isMissingColumn())
    {
        return;
    }
    auto &&fieldPtr = m_structMeta -> field( fieldName );
    CSP_TRUE_OR_THROW_RUNTIME( fieldPtr != nullptr,
                               "No field " << fieldName << " in struct " << m_structMeta -> name() );

    FieldSetter fieldSetter;
    if( fieldPtr -> type() -> type() == CspType::TypeTraits::ENUM )
    {
        columnAdapter.ensureType( CspType::STRING() );
        auto enumMetaPtr = std::static_pointer_cast<const CspEnumType>( fieldPtr -> type() ) -> meta();
        using ColumnType = CspType::Type::toCType<CspType::Type::STRING>::type;
        fieldSetter = [ &columnAdapter, fieldPtr, enumMetaPtr ]( StructPtr &s )
        {
            auto curValue = columnAdapter.getCurValue<ColumnType>();
            if( curValue.has_value() )
            {
                fieldPtr -> setValue<CspEnum>( s.get(), enumMetaPtr -> fromString( curValue.value().c_str() ) );
            }
        };
    }
    else
    {
        using TypeSwitch = PrimitiveCspTypeSwitch::Extend<CspType::Type::STRUCT>;
        columnAdapter.ensureType( fieldPtr -> type() );
        if( fieldPtr -> type() -> type() == CspType::Type::STRUCT )
        {
            auto &fieldMeta = std::static_pointer_cast<const CspStructType>( fieldPtr -> type() ) -> meta();
            static_cast<StructColumnAdapter &>(columnAdapter).initFromStructMeta( fieldMeta );
        }
        if( columnAdapter.isNativeType() )
        {
            fieldSetter = NativeCspTypeSwitch::invoke(
                    ( columnAdapter.getNativeCspType().get() ), [ &columnAdapter, &fieldPtr ]( auto columnTag )
                    {
                        using ColType = typename decltype(columnTag)::type;
                        return ConstructibleTypeSwitch<ColType>::invoke(
                                fieldPtr -> type().get(),
                                [ &columnAdapter, &fieldPtr ]( auto tag )
                                {
                                    using FieldType = typename decltype(tag)::type;
                                    return FieldSetter( [ &columnAdapter, fieldPtr ]( StructPtr &s )
                                                        {
                                                            auto curValue = columnAdapter.getCurValue<ColType>();
                                                            if( curValue.has_value() )
                                                            {
                                                                fieldPtr -> setValue<FieldType>( s.get(), csp::cast<FieldType>(
                                                                        curValue.value() ) );
                                                            }
                                                        } );
                                } );
                    }

            );
        }
        else
        {
            fieldSetter = TypeSwitch::invoke( fieldPtr -> type().get(), [ &columnAdapter, fieldPtr ]( auto tag )
            {
                using FieldType = typename decltype(tag)::type;
                return FieldSetter( [ &columnAdapter, fieldPtr ]( StructPtr &s )
                                    {
                                        auto curValue = columnAdapter.getCurValue<FieldType>();
                                        if( curValue.has_value() )
                                        {
                                            fieldPtr -> setValue<FieldType>( s.get(), curValue.value() );
                                        }
                                    } );
            } );
        }
    }
    m_fieldSetters.push_back( fieldSetter );
}

void ParquetStructAdapter::addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol )
{
    CSP_TRUE_OR_THROW( inputAdapter -> type() -> type() == CspType::Type::STRUCT, TypeError,
                       "Subscribing unexpected type " << inputAdapter -> type() -> type() << " as struct for column " );
    auto meta = static_cast<const CspStructType *>( inputAdapter -> type()) -> meta();
    CSP_TRUE_OR_THROW( meta == m_structMeta, TypeError,
                       "Subscribing " << meta -> name() << " where " << m_structMeta -> name() << " is expected" );

    m_valueDispatcher.addSubscriber( [ inputAdapter ]( StructPtr *s )
                                       {
                                           if( s )
                                           {
                                               inputAdapter -> pushTick( *s );
                                           }
                                           else
                                           {
                                               inputAdapter -> pushNullTick<StructPtr>();
                                           }
                                       },
                                       symbol );

}

void ParquetStructAdapter::addSubscriber( ValueDispatcher::SubscriberType subscriber, std::optional<utils::Symbol> symbol )
{
    m_valueDispatcher.addSubscriber( subscriber, symbol );
}

void ParquetStructAdapter::dispatchValue( const utils::Symbol *symbol, bool isNull )
{
    if(unlikely(m_needsReset))
    {
        m_resetFunc();
        m_needsReset = false;
    }

    auto allSymbolSubscribers = m_valueDispatcher.getSubscribers();
    auto symbolSubscribers    = symbol != nullptr ? m_valueDispatcher.getSubscribersForSymbol( *symbol ) : nullptr;

    if( allSymbolSubscribers == nullptr && symbolSubscribers == nullptr )
    {
        return;
    }

    StructPtr s;
    StructPtr *dispatchedValue;
    if( isNull )
    {
        dispatchedValue = nullptr;
    }
    else
    {
        s = StructPtr{ m_structMeta -> create() };
        for( auto &fieldSetter : m_fieldSetters )
        {
            fieldSetter( s );
        }
        dispatchedValue = &s;
    }

    if( allSymbolSubscribers )
    {
        m_valueDispatcher.dispatch( dispatchedValue, *allSymbolSubscribers );
    }
    if( symbolSubscribers )
    {
        m_valueDispatcher.dispatch( dispatchedValue, *symbolSubscribers );
    }
}

template< typename ValueType, typename ArrowArrayType >
void NativeTypeColumnAdapter<ValueType, ArrowArrayType>::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = csp::cast<ValueType>( this -> m_curChunkArray -> Value( curRow ) );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

template< int64_t UNIT >
void DatetimeColumnAdapter<UNIT>::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = csp::DateTime::fromNanoseconds(
                m_curChunkArray -> Value( m_parquetReader.getCurRow() ) * UNIT );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

template< int64_t UNIT >
void DurationColumnAdapter<UNIT>::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = csp::TimeDelta::fromNanoseconds(
                m_curChunkArray -> Value( m_parquetReader.getCurRow() ) * UNIT );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

template< int64_t UNIT, typename ArrowDateArray >
void DateColumnAdapter<UNIT, ArrowDateArray>::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = csp::DateTime::fromNanoseconds(
                this -> m_curChunkArray -> Value( this -> m_parquetReader.getCurRow() ) * UNIT ).date();
    }
    else
    {
        this -> m_curValue.reset();
    }
}

template< int64_t UNIT, typename ArrowTimeArray >
void TimeColumnAdapter<UNIT, ArrowTimeArray>::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = csp::Time::fromNanoseconds(
            this -> m_curChunkArray -> Value( this -> m_parquetReader.getCurRow() ) * UNIT );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

template< typename ArrowStringArrayType>
void StringColumnAdapter<ArrowStringArrayType>::addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol )
{
    if( inputAdapter -> type() -> type() == CspType::TypeTraits::ENUM )
    {
        auto enumMetaPtr = static_cast<const CspEnumType *>( inputAdapter -> type()) -> meta();
        auto callback    = std::function<void(
                const std::string * )>(
                [ inputAdapter, enumMetaPtr ]( const std::string *val )
                {
                    if( val )
                    {
                        inputAdapter -> pushTick<CspEnum>( enumMetaPtr -> fromString( val -> c_str() ) );
                    }
                    else
                    {
                        inputAdapter -> pushNullTick<CspEnum>();
                    }
                } );
        m_dispatcher.addSubscriber( callback, symbol );
    }
    else
    {
        BaseTypedColumnAdapter<std::string, ArrowStringArrayType>::addSubscriber( inputAdapter, symbol );
    }
}

template< typename ArrowStringArrayType>
void StringColumnAdapter<ArrowStringArrayType>::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = this -> m_curChunkArray -> GetString( curRow );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

template< typename ArrowBytesArrayType >
void BytesColumnAdapter<ArrowBytesArrayType>::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = this -> m_curChunkArray -> GetString( curRow );
    }
    else
    {
        this -> m_curValue.reset();
    }
}


void FixedSizeBinaryColumnAdapter::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        this -> m_curValue = this -> m_curChunkArray -> GetString( curRow );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

void DictionaryColumnAdapter::readCurValue()
{
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        auto index = this -> m_curChunkArray -> GetValueIndex( curRow );
        auto dict  = static_cast<arrow::StringArray *>(this -> m_curChunkArray -> dictionary().get());
        this -> m_curValue = dict -> GetString( index );
    }
    else
    {
        this -> m_curValue.reset();
    }
}


namespace
{
    template<typename T>
    struct ArrayValidValueProvider
    {
        template<typename V>
        static T getValue(const V& array, int index)
        {
            if(!array->IsValid(index))
            {
                CSP_THROW(ValueError, "Can't read empty value to array from arrow array of type " << arrow::StringArray::TypeClass::type_name());
            }
            return array->GetView(index);
        }
    };

    template<>
    struct ArrayValidValueProvider<double>
    {
        template<typename V>
        static double getValue(const V& array, int index)
        {
            if(!array->IsValid(index))
            {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return std::move(array->GetView(index));
        }
    };
}

template< typename ArrowListArrayType, typename ValueArrayType, typename ValueType>
void BaseListColumnAdapter<ArrowListArrayType, ValueArrayType, ValueType>::addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol )
{
    CSP_THROW( NotImplemented, "Trying to subscribe to list column indirectly, via struct field?" );
}

template< typename ArrowListArrayType, typename ValueArrayType, typename ValueType>
void BaseListColumnAdapter<ArrowListArrayType, ValueArrayType, ValueType>::addSubscriber( ManagedSimInputAdapter *inputAdapter,
                    std::optional<utils::Symbol> symbol, const DialectGenericListReaderInterface::Ptr &listReader )
{
    CSP_TRUE_OR_THROW_RUNTIME( m_listReader == nullptr,
                               "Trying to subscribe list column in parquet reader more than once, this is not supported" );
    CSP_TRUE_OR_THROW_RUNTIME( listReader != nullptr,
                               "Trying to subscribe list column in parquet reader with null listReader" );
    BaseTypedColumnAdapter<DialectGenericType, ArrowListArrayType>::addSubscriber( inputAdapter, symbol );

    m_listReader = std::dynamic_pointer_cast<TypedDialectGenericListReaderInterface<ValueType>>( listReader );
    CSP_TRUE_OR_THROW_RUNTIME( m_listReader != nullptr,
                               "Subscribed to parquet column " << getColumnName() << " with type "
                                                               << "NumpyArray[" << listReader -> getValueType() -> type().asString()
                                                               << "] while "
                                                               << " column type in file is NumpyArray["
                                                               << getContainerValueType() -> type().asString() << "]"
                                                               << " in file " << m_parquetReader.getCurFileOrTableName() );
}

template< typename ArrowListArrayType, typename ValueArrayType, typename ValueType >
void NativeListColumnAdapter<ArrowListArrayType, ValueArrayType, ValueType>::readCurValue()
{
    CSP_TRUE_OR_THROW_RUNTIME( m_listReader != nullptr,
                               "Trying to read list value from parquet file but not list reader interface is set" );

    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        auto values      = this -> m_curChunkArray -> value_slice( curRow );
        auto typedValues = std::dynamic_pointer_cast<ValueArrayType>( values );

        auto arrayValue     = m_listReader -> create( typedValues -> length() );
        auto* internalBuffer = m_listReader -> getRawDataBuffer( arrayValue );
        if( internalBuffer != nullptr )
        {
            for( int64_t i = 0; i < typedValues -> length(); ++i )
            {
                *(internalBuffer++) = ArrayValidValueProvider<ValueType>::getValue(typedValues, i);
            }
        }
        else
        {
            for( int64_t i = 0; i < typedValues -> length(); ++i )
            {
                m_listReader -> setValue( arrayValue, i, ArrayValidValueProvider<ValueType>::getValue(typedValues, i) );
            }
        }
        this -> m_curValue = std::move( arrayValue );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

template< typename ArrowListArrayType, typename ArrowBytesArrayType >
void BytesListColumnAdapter<ArrowListArrayType, ArrowBytesArrayType >::readCurValue()
{
    CSP_TRUE_OR_THROW_RUNTIME( m_listReader != nullptr,
                               "Trying to read list value from parquet file but not list reader interface is set" );

    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        auto values      = this -> m_curChunkArray -> value_slice( curRow );
        auto typedValues = std::dynamic_pointer_cast<ArrowBytesArrayType>( values );

        uint32_t maxStringLength = 0;
        for( int64_t i = 0; i < typedValues -> length(); ++i )
        {

            maxStringLength = std::max( ( uint32_t ) ArrayValidValueProvider<std::string_view>::getValue(typedValues, i).length(), maxStringLength );
        }

        auto arrayValue = m_listReader -> create( typedValues -> length(), maxStringLength );

        for( int64_t i = 0; i < typedValues -> length(); ++i )
        {
            m_listReader -> setValue( arrayValue, i, std::string( typedValues -> GetView( i ) ) );
        }
        this -> m_curValue = std::move( arrayValue );
    }
    else
    {
        this -> m_curValue.reset();
    }
}

void StructColumnAdapter::addSubscriber( ManagedSimInputAdapter *inputAdapter, std::optional<utils::Symbol> symbol )
{
    CSP_TRUE_OR_THROW_RUNTIME( inputAdapter -> type() -> type() == CspType::TypeTraits::STRUCT,
                               "Trying to subscribe with non struct type " << inputAdapter -> type() -> type().asString() );
    auto structMeta = static_cast<const CspStructType *>( inputAdapter -> type() ) -> meta();
    initFromStructMeta( structMeta );
    BASE::addSubscriber( inputAdapter, symbol );
}

void StructColumnAdapter::readCurValue()
{
    // TODO: In the future we should make optimization here and skip creating structs for symbols for which there's no subscription
    auto curRow = this -> m_parquetReader.getCurRow();
    if( this -> m_curChunkArray -> IsValid( curRow ) )
    {
        for( auto &childColumnAdapter: m_childColumnAdapters )
        {
            childColumnAdapter -> readCurValue();
        }
        m_structAdapter -> dispatchValue( nullptr );
    }
    else
    {
        m_structAdapter -> dispatchValue( nullptr, true );
        this -> m_curValue.reset();
    }
}

void StructColumnAdapter::handleNewBatch( const std::shared_ptr<::arrow::ChunkedArray> &data )
{
    BASE::handleNewBatch( data );
    auto &childArrays = m_curChunkArray -> fields();
    CSP_TRUE_OR_THROW_RUNTIME( childArrays.size() == m_childColumnAdapters.size(),
                               "Expected " << m_childColumnAdapters.size() << " child arrays, got " << childArrays.size() );

    for( std::size_t i = 0; i < childArrays.size(); ++i )
    {
        m_childColumnAdapters[ i ] -> handleNewBatch( childArrays[ i ] );
    }
}

void StructColumnAdapter::handleNewBatch( const std::shared_ptr<::arrow::Array> &data )
{
    BASE::handleNewBatch( data );
    auto &childArrays = m_curChunkArray -> fields();
    CSP_TRUE_OR_THROW_RUNTIME( childArrays.size() == m_childColumnAdapters.size(),
                               "Expected " << m_childColumnAdapters.size() << " child arrays, got " << childArrays.size() );

    for( std::size_t i = 0; i < childArrays.size(); ++i )
    {
        m_childColumnAdapters[ i ] -> handleNewBatch( childArrays[ i ] );
    }
}

void StructColumnAdapter::initFromStructMeta( const std::shared_ptr<StructMeta> &structMeta )
{
    if( m_structAdapter )
    {
        CSP_TRUE_OR_THROW_RUNTIME( m_structAdapter -> getStructMeta() == structMeta,
                                   "Trying to subscribe to structure field with struct "
                                           << structMeta -> name() << " and " << m_structAdapter -> getStructMeta() -> name() );
        return;
    }

    m_childColumnAdapters.reserve( m_arrowType -> num_fields() );
    for( auto &field:m_arrowType -> fields() )
    {
        m_childColumnAdapters.push_back( createColumnAdapter( m_parquetReader, *field, "" ) );
    }
    m_structAdapter = std::make_unique<ParquetStructAdapter>( m_parquetReader,
                                                              m_arrowType,
                                                              structMeta,
                                                              m_childColumnAdapters );
    m_structAdapter -> addSubscriber( [ this ]( StructPtr *s )
                                      {
                                          if( s )
                                          {
                                              this -> m_curValue = *s;
                                          }
                                      } );
}

}
