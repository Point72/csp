#include <csp/adapters/parquet/ParquetInputAdapterManager.h>
#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <csp/adapters/arrow/RecordBatchRowProcessor.h>
#include <csp/adapters/arrow/ArrowTypeVisitor.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/TypeCast.h>
#include <csp/engine/PartialSwitchCspType.h>

namespace csp::adapters::parquet
{

uint16_t ParquetInputAdapterManager::DictBasketReaderRecord::getValueCount() const
{
    // After a schema change the new schema may lack the value_count column
    if( !m_valueCountDispatcher )
        return 0;
    auto opt = m_valueCountDispatcher -> getCurValue<uint16_t>();
    CSP_TRUE_OR_THROW_RUNTIME( opt.has_value(), "Null value in dict basket value count column" );
    return opt.value();
}

namespace
{

using FieldSetter = std::function<void( StructPtr & )>;

template<typename ColType>
FieldSetter makeFieldSetter( const StructFieldPtr & fieldPtr, arrow::ColumnDispatcher & dispatcher )
{
    using TypeSwitch = ConstructibleTypeSwitch<ColType>;
    arrow::ColumnDispatcher * dispPtr = &dispatcher;
    StructField * rawField = fieldPtr.get();
    return TypeSwitch::invoke( fieldPtr -> type().get(),
        [dispPtr, rawField]( auto tag )
        {
            using FieldType = typename decltype(tag)::type;
            return FieldSetter( [dispPtr, rawField]( StructPtr & s )
            {
                auto & val = dispPtr -> getCurValue<ColType>();
                if( val.has_value() )
                    rawField -> setValue<FieldType>( s.get(), csp::cast<FieldType>( val.value() ) );
            } );
        } );
}

FieldSetter createFieldSetterForDispatcher( const StructFieldPtr & fieldPtr,
                                            arrow::ColumnDispatcher & dispatcher )
{
    arrow::ColumnDispatcher * dispPtr = &dispatcher;

    // Enum fields: string → CspEnum coercion
    if( fieldPtr -> type() -> type() == CspType::Type::ENUM )
    {
        StructField * rawField = fieldPtr.get();
        auto * rawEnumMeta = std::static_pointer_cast<const CspEnumType>( fieldPtr -> type() ) -> meta().get();
        return FieldSetter( [dispPtr, rawField, rawEnumMeta]( StructPtr & s )
        {
            auto & val = dispPtr -> getCurValue<std::string>();
            if( val.has_value() )
                rawField -> setValue<CspEnum>( s.get(), rawEnumMeta -> fromString( val.value().c_str() ) );
        } );
    }

    return arrow::visitArrowValueType( dispatcher.arrowTypeId(),
        [&]( auto tag ) -> FieldSetter
        {
            using T = typename decltype( tag )::type;
            return makeFieldSetter<T>( fieldPtr, dispatcher );
        },
        [&]() -> FieldSetter
        {
            CSP_THROW( TypeError, "Unsupported arrow type for struct field '" << fieldPtr -> fieldname() << "'" );
        } );
}

// Check if arrow-mapped C++ type T is compatible with a CspType.
template<typename T>
bool isArrowTypeCompatible( CspType::Type t )
{
    using CT = CspType::Type;
    if constexpr( std::is_same_v<T, bool> )
        return t == CT::BOOL;
    else if constexpr( std::is_integral_v<T> )
        return t == CT::INT8 || t == CT::UINT8 || t == CT::INT16 || t == CT::UINT16 ||
               t == CT::INT32 || t == CT::UINT32 || t == CT::INT64 || t == CT::UINT64 ||
               t == CT::DOUBLE;
    else if constexpr( std::is_same_v<T, double> )
        return t == CT::DOUBLE;
    else if constexpr( std::is_same_v<T, std::string> )
        return t == CT::STRING || t == CT::ENUM;
    else if constexpr( std::is_same_v<T, DateTime> )        return t == CT::DATETIME;
    else if constexpr( std::is_same_v<T, TimeDelta> )       return t == CT::TIMEDELTA;
    else if constexpr( std::is_same_v<T, Date> )            return t == CT::DATE;
    else if constexpr( std::is_same_v<T, Time> )            return t == CT::TIME;
    else if constexpr( std::is_same_v<T, DialectGenericType> ) return t == CT::DIALECT_GENERIC;
    else if constexpr( std::is_same_v<T, StructPtr> )       return t == CT::STRUCT;
    else return false;
}

} // anonymous namespace

// Validate arrow type is compatible with subscriber's CspType
static void ensureType( const std::string & columnName, ::arrow::Type::type arrowType, const std::string & arrowTypeName,
                        const CspType * cspType )
{
    auto t = cspType -> type();
    bool compatible = arrow::visitArrowValueType( arrowType,
        [t]( auto tag ) -> bool { return isArrowTypeCompatible<typename decltype( tag )::type>( t ); },
        [&]() -> bool { return false; } );

    CSP_TRUE_OR_THROW( compatible, TypeError,
        "Unexpected column type for column " << columnName << " , expected " << t.asCString()
        << " got " << arrowTypeName );
}


void ParquetInputAdapterManager::StructSubscription::createFieldSetters(
    arrow::RecordBatchRowProcessor & processor,
    const std::shared_ptr<::arrow::Schema> & schema )
{
    m_fieldSetters.clear();
    for( auto it = m_fieldMap -> begin(); it != m_fieldMap -> end(); ++it )
    {
        auto & parquetColumnName = it.key();
        auto & structFieldName   = it.value<std::string>();
        auto * dispatcher        = processor.getDispatcher( parquetColumnName );
        if( !dispatcher )
            continue;

        auto fieldPtr = m_structMeta -> field( structFieldName );
        CSP_TRUE_OR_THROW_RUNTIME( fieldPtr != nullptr,
            "No field '" << structFieldName << "' in struct " << m_structMeta -> name() );

        if( fieldPtr -> type() -> type() == CspType::Type::STRUCT &&
            dispatcher -> arrowTypeId() != ::arrow::Type::STRUCT )
        {
            continue;
        }

        m_fieldSetters.push_back( createFieldSetterForDispatcher( fieldPtr, *dispatcher ) );
    }
}

void ParquetInputAdapterManager::StructSubscription::dispatchValue( const utils::Symbol * symbol )
{
    auto allSymbolSubscribers = m_valueDispatcher.getSubscribers();
    auto symbolSubscribers    = symbol != nullptr ? m_valueDispatcher.getSubscribersForSymbol( *symbol ) : nullptr;

    if( allSymbolSubscribers == nullptr && symbolSubscribers == nullptr )
        return;

    StructPtr s{ m_structMeta -> create() };
    for( auto & setter : m_fieldSetters )
        setter( s );

    if( allSymbolSubscribers )
        m_valueDispatcher.dispatch( &s, *allSymbolSubscribers );
    if( symbolSubscribers )
        m_valueDispatcher.dispatch( &s, *symbolSubscribers );
}

void ParquetInputAdapterManager::StructSubscription::addSubscriber(
    ManagedSimInputAdapter * adapter, std::optional<utils::Symbol> symbol )
{
    CSP_TRUE_OR_THROW( adapter -> type() -> type() == CspType::Type::STRUCT, TypeError,
        "Subscribing unexpected type " << adapter -> type() -> type() << " as struct" );
    auto meta = static_cast<const CspStructType *>( adapter -> type() ) -> meta();
    CSP_TRUE_OR_THROW( meta == m_structMeta, TypeError,
        "Subscribing " << meta -> name() << " where " << m_structMeta -> name() << " is expected" );

    m_valueDispatcher.addSubscriber( [ adapter ]( StructPtr * s )
    {
        if( s )
            adapter -> pushTick( *s );
        else
            adapter -> pushNullTick<StructPtr>();
    }, symbol );
}


ParquetInputAdapterManager::ParquetInputAdapterManager( csp::Engine *engine, const Dictionary &properties,
                                                        RecordBatchStreamSourcePtr streamSource) :
        AdapterManager( engine ),
        m_time_shift(0, 0),
        m_streamSource(streamSource)
{
    CSP_TRUE_OR_THROW_RUNTIME( m_streamSource, "RecordBatch stream source must be provided" );

    m_symbolColumn        = properties.get<std::string>( "symbol_column", "" );
    m_timeColumn          = properties.get<std::string>( "time_column", "" );
    m_defaultTimezone     = properties.get<std::string>( "tz", "UTC" );
    m_allowOverlappingPeriods = properties.get<bool>( "allow_overlapping_periods", false );
    m_allowMissingColumns = properties.get<bool>( "allow_missing_columns", false );
    properties.tryGet( "start_time", m_startTime );
    properties.tryGet( "end_time", m_endTime );
    properties.tryGet( "time_shift", m_time_shift );

    CSP_TRUE_OR_THROW_RUNTIME( m_timeColumn != "", "Time column can't be empty" );
    CSP_TRUE_OR_THROW_RUNTIME( m_defaultTimezone == "UTC",
                               "Only UTC default timezone is supported, got:" << m_defaultTimezone );
}

ParquetInputAdapterManager::~ParquetInputAdapterManager() = default;

void ParquetInputAdapterManager::collectAdapterColumns(
    const AdaptersBySymbol & adaptersBySymbol,
    std::set<std::string> & columns )
{
    for( auto && adaptersForSymbol : adaptersBySymbol )
    {
        for( auto && entry : adaptersForSymbol.second.m_adaptersByColumnName )
            columns.insert( entry.first );

        for( auto && entry : adaptersForSymbol.second.m_structAdapters )
        {
            auto && fieldMap = entry.first.fieldMap();
            for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
                columns.insert( it.key() );
        }
    }
}

void ParquetInputAdapterManager::setupProcessor(
    arrow::RecordBatchRowProcessor & processor,
    const std::shared_ptr<::arrow::Schema> & schema,
    const std::set<std::string> & neededColumns,
    const AdaptersBySymbol & adaptersBySymbol,
    bool subscribeAllOnEmptySymbol )
{
    std::set<std::string> allColumns{ neededColumns };
    collectAdapterColumns( adaptersBySymbol, allColumns );

    // Collect StructMeta for STRUCT columns so setupFromSchema can create nested dispatchers.
    std::unordered_map<std::string, std::shared_ptr<const StructMeta>> structMetaByColumn;

    for( auto && adaptersForSymbol : adaptersBySymbol )
    {
        for( auto && entry : adaptersForSymbol.second.m_adaptersByColumnName )
        {
            auto * adapterType = entry.second.m_adapter -> dataType();
            if( adapterType -> type() == CspType::Type::STRUCT )
            {
                auto meta = static_cast<const CspStructType *>( adapterType ) -> meta();
                auto [it2, inserted] = structMetaByColumn.emplace( entry.first, meta );
                if( !inserted && it2 -> second != meta )
                    CSP_THROW( TypeError, "Conflicting struct types for column '" << entry.first
                        << "': multiple subscriptions use different struct types for the same STRUCT column" );
            }
        }

        for( auto && entry : adaptersForSymbol.second.m_structAdapters )
        {
            auto parentMeta = std::static_pointer_cast<const CspStructType>( entry.first.type() ) -> meta();
            auto && fieldMap = entry.first.fieldMap();
            for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
            {
                auto & structFieldName = it.value<std::string>();
                auto fieldPtr = parentMeta -> field( structFieldName );
                if( fieldPtr && fieldPtr -> type() -> type() == CspType::Type::STRUCT )
                {
                    auto meta = std::static_pointer_cast<const CspStructType>( fieldPtr -> type() ) -> meta();
                    auto [it2, inserted] = structMetaByColumn.emplace( it.key(), meta );
                    if( !inserted && it2 -> second != meta )
                        CSP_THROW( TypeError, "Conflicting struct types for column '" << it.key()
                            << "': multiple subscriptions use different struct types for the same STRUCT column" );
                }
            }
        }
    }

    processor.setupFromSchema( schema, allColumns, m_allowMissingColumns, structMetaByColumn );
}

void ParquetInputAdapterManager::subscribeAdapters(
    arrow::RecordBatchRowProcessor & processor,
    const AdaptersBySymbol & adaptersBySymbol,
    bool subscribeAllOnEmptySymbol,
    std::vector<std::unique_ptr<StructSubscription>> & structSubscriptions )
{
    for( auto & symMapPair : adaptersBySymbol )
    {
        std::optional<utils::Symbol> symbol;
        if( !subscribeAllOnEmptySymbol ||
            !std::holds_alternative<std::string>( symMapPair.first ) ||
            ( std::holds_alternative<std::string>( symMapPair.first ) &&
              std::get<std::string>( symMapPair.first ) != "" ) )
        {
            symbol = symMapPair.first;
        }

        // Validate symbol type if subscriber specified one
        if( symbol.has_value() && !m_symbolColumn.empty() )
        {
            if( m_symbolType == CspType::Type::STRING )
                CSP_TRUE_OR_THROW( std::holds_alternative<std::string>( symbol.value() ), TypeError,
                    "Provided symbol type does not match symbol column type (string)" );
            else if( m_symbolType == CspType::Type::INT64 )
                CSP_TRUE_OR_THROW( std::holds_alternative<int64_t>( symbol.value() ), TypeError,
                    "Provided symbol type does not match symbol column type (int64)" );
        }

        // Scalar/list subscriptions
        for( auto & columnPair : symMapPair.second.m_adaptersByColumnName )
        {
            if( !processor.hasColumn( columnPair.first ) )
                continue;

            auto * dispatcher = processor.getDispatcher( columnPair.first );
            bool isList = ( dispatcher -> arrowTypeId() == ::arrow::Type::LIST ||
                            dispatcher -> arrowTypeId() == ::arrow::Type::LARGE_LIST );
            bool isListAdapter = ( columnPair.second.m_adapter -> dataType() -> type() == CspType::Type::DIALECT_GENERIC );

            CSP_TRUE_OR_THROW_RUNTIME( !isList || isListAdapter,
                "Column " << columnPair.first << " is a list column while subscribing as non-list" );
            CSP_TRUE_OR_THROW_RUNTIME( isList || !isListAdapter,
                "Column " << columnPair.first << " is not a list column while subscribing as list" );

            auto * adapterCspType = columnPair.second.m_adapter -> dataType();
            int colIdx = m_curSchema -> GetFieldIndex( columnPair.first );
            if( colIdx >= 0 )
            {
                auto arrowField = m_curSchema -> field( colIdx );
                ensureType( columnPair.first, dispatcher -> arrowTypeId(),
                            arrowField -> type() -> name(), adapterCspType );
            }

            processor.addSubscriber( columnPair.first, columnPair.second.m_adapter, symbol );
        }

        // Struct subscriptions
        for( auto & structPair : symMapPair.second.m_structAdapters )
        {
            auto structMeta = std::static_pointer_cast<const CspStructType>( structPair.first.type() ) -> meta();

            StructSubscription * sub = nullptr;
            for( auto & existing : structSubscriptions )
            {
                if( existing -> m_structMeta == structMeta && existing -> m_fieldMap == structPair.first.fieldMap() )
                {
                    sub = existing.get();
                    break;
                }
            }
            if( !sub )
            {
                auto newSub = std::make_unique<StructSubscription>();
                newSub -> m_structMeta = structMeta;
                newSub -> m_fieldMap   = structPair.first.fieldMap();
                sub = newSub.get();
                structSubscriptions.push_back( std::move( newSub ) );
            }

            sub -> addSubscriber( structPair.second.m_adapter, symbol );
        }
    }

    for( auto & sub : structSubscriptions )
        sub -> createFieldSetters( processor, m_curSchema );
}

void ParquetInputAdapterManager::setupBasketProcessor(
    DictBasketReaderRecord & record,
    const AdaptersBySymbol & adaptersBySymbol )
{
    std::string symCol = record.m_basketName + "__csp_symbol";
    std::set<std::string> basketNeeded{ symCol };

    setupProcessor( *record.m_processor, m_curSchema, basketNeeded, adaptersBySymbol, false );
    record.m_cachedSymbolDispatcher = record.m_processor -> getDispatcher( symCol );

    record.m_structSubscriptions.clear();
    subscribeAdapters( *record.m_processor, adaptersBySymbol, false, record.m_structSubscriptions );
}

void ParquetInputAdapterManager::setupDictBaskets()
{
    for( auto && it : m_dictBasketInputAdapters )
    {
        auto valueCountColumnName = it.first + "__csp_value_count";

        DictBasketReaderRecord record;
        record.m_basketName = it.first;
        record.m_basketSymbolColumn = it.first + "__csp_symbol";
        record.m_valueCountDispatcher = m_processor -> getDispatcher( valueCountColumnName );
        CSP_TRUE_OR_THROW_RUNTIME( record.m_valueCountDispatcher != nullptr,
            "Value count column '" << valueCountColumnName << "' not found" );

        record.m_processor = std::make_unique<arrow::RecordBatchRowProcessor>();
        setupBasketProcessor( record, it.second );

        m_dictBasketReaders.push_back( std::move( record ) );
    }
}

std::shared_ptr<::arrow::Schema> ParquetInputAdapterManager::buildLogicalSchema(
    const ColumnReaderMap & readers )
{
    std::vector<std::shared_ptr<::arrow::Field>> fields;
    for( auto & [colName, reader] : readers )
    {
        auto readerSchema = reader -> schema();
        for( int i = 0; i < readerSchema -> num_fields(); ++i )
            fields.push_back( readerSchema -> field( i ) );
    }
    return ::arrow::schema( fields );
}

bool ParquetInputAdapterManager::bindSourcesFromReaders()
{
    m_mainRBSources.clear();

    // Build a lookup from basket name → DictBasketReaderRecord
    std::unordered_map<std::string, DictBasketReaderRecord *> basketByName;
    for( auto & record : m_dictBasketReaders )
        basketByName[ record.m_basketName ] = &record;

    // Collect sources and mappings for main processor
    std::vector<::arrow::RecordBatchReader *> mainSources;
    std::vector<std::vector<arrow::RecordBatchRowProcessor::ColumnMapping>> mainMappings;

    // Per-basket: collect sources and mappings
    std::unordered_map<std::string, std::vector<::arrow::RecordBatchReader *>> basketSources;
    std::unordered_map<std::string, std::vector<std::vector<arrow::RecordBatchRowProcessor::ColumnMapping>>> basketMappings;

    auto & readers = m_streamSource -> columnReaders();
    for( auto & [dictKey, reader] : readers )
    {
        auto readerSchema = reader -> schema();

        // Separate columns into main vs basket
        std::vector<arrow::RecordBatchRowProcessor::ColumnMapping> mainCols;
        std::unordered_map<std::string, std::vector<arrow::RecordBatchRowProcessor::ColumnMapping>> basketCols;

        for( int i = 0; i < readerSchema -> num_fields(); ++i )
        {
            auto colName = readerSchema -> field( i ) -> name();

            auto basketIt = m_columnToBasketName.find( colName );
            if( basketIt == m_columnToBasketName.end() )
            {
                // Main column
                if( m_processor -> hasColumn( colName ) )
                    mainCols.push_back( { colName, i } );
            }
            else
            {
                // Basket column
                auto recordIt = basketByName.find( basketIt -> second );
                if( recordIt != basketByName.end() && recordIt -> second -> m_processor -> hasColumn( colName ) )
                    basketCols[ basketIt -> second ].push_back( { colName, i } );
            }
        }

        if( !mainCols.empty() )
        {
            mainSources.push_back( reader.get() );
            mainMappings.push_back( std::move( mainCols ) );
        }

        for( auto & [bname, cols] : basketCols )
        {
            basketSources[ bname ].push_back( reader.get() );
            basketMappings[ bname ].push_back( std::move( cols ) );
        }

        m_mainRBSources.push_back( reader );
    }

    // Bind main processor
    if( mainSources.empty() )
        return false;
    m_processor -> bindSources( mainSources, mainMappings );

    // Bind basket processors
    for( auto & record : m_dictBasketReaders )
    {
        auto srcIt = basketSources.find( record.m_basketName );
        if( srcIt != basketSources.end() )
        {
            auto mapIt = basketMappings.find( record.m_basketName );
            record.m_processor -> bindSources( srcIt -> second, mapIt -> second );
        }
        else
        {
            record.m_processor -> bindSources( {}, {} );
        }
    }

    return true;
}

bool ParquetInputAdapterManager::advanceToNextStream()
{
    while( m_streamSource -> nextStream() )
    {
        auto newSchema = buildLogicalSchema( m_streamSource -> columnReaders() );
        bool schemaChanged = m_curSchema && !m_curSchema -> Equals( *newSchema );
        m_curSchema = newSchema;

        if( schemaChanged )
        {
            m_structSubscriptions.clear();
            setupProcessor( *m_processor, m_curSchema, m_neededColumns, m_simInputAdapters, true );

            // Re-detect symbol type — the new schema may have a different arrow type.
            if( !m_symbolColumn.empty() )
            {
                auto * symDispatcher = m_processor -> getDispatcher( m_symbolColumn );
                if( symDispatcher )
                    detectSymbolType( symDispatcher );
                else
                    m_cachedSymbolDispatcher = nullptr;
            }

            subscribeAdapters( *m_processor, m_simInputAdapters, true, m_structSubscriptions );

            m_cachedTimeDispatcher = m_processor -> getDispatcher( m_timeColumn );
            CSP_TRUE_OR_THROW_RUNTIME( m_cachedTimeDispatcher != nullptr,
                "Time column '" << m_timeColumn << "' not found after schema change" );

            for( auto & record : m_dictBasketReaders )
            {
                auto adapterIt = m_dictBasketInputAdapters.find( record.m_basketName );
                if( adapterIt == m_dictBasketInputAdapters.end() )
                    continue;
                auto vcName = record.m_basketName + "__csp_value_count";
                record.m_valueCountDispatcher = m_processor -> getDispatcher( vcName );
                setupBasketProcessor( record, adapterIt -> second );
            }
        }

        if( !bindSourcesFromReaders() )
            continue;  // empty stream — try next

        if( m_processor -> readRowAndAdvance() )
            return true;
    }
    return false;
}

bool ParquetInputAdapterManager::readNextRow()
{
    if( !m_hasData ) [[unlikely]]
        return false;

    if( m_processor -> readRowAndAdvance() )
        return true;

    if( !advanceToNextStream() )
    {
        m_hasData = false;
        return false;
    }
    return true;
}

static bool extractSymbolFromDispatcher( arrow::ColumnDispatcher * dispatcher, utils::Symbol & out )
{
    auto symType = dispatcher -> arrowTypeId();
    if( symType == ::arrow::Type::STRING || symType == ::arrow::Type::LARGE_STRING || symType == ::arrow::Type::DICTIONARY )
    {
        auto & val = dispatcher -> getCurValue<std::string>();
        if( val.has_value() ) { out = val.value(); return true; }
    }
    else if( symType == ::arrow::Type::INT64 )
    {
        auto & val = dispatcher -> getCurValue<int64_t>();
        if( val.has_value() ) { out = val.value(); return true; }
    }
    return false;
}

void ParquetInputAdapterManager::detectSymbolType( arrow::ColumnDispatcher * symDispatcher )
{
    auto symType = symDispatcher -> arrowTypeId();
    if( symType == ::arrow::Type::STRING || symType == ::arrow::Type::LARGE_STRING ||
        symType == ::arrow::Type::DICTIONARY )
        m_symbolType = CspType::Type::STRING;
    else if( symType == ::arrow::Type::INT64 )
        m_symbolType = CspType::Type::INT64;
    else
        CSP_THROW( TypeError, "Invalid symbol column type. Only string and int64 symbols are currently supported" );
    m_cachedSymbolDispatcher = symDispatcher;
}

const utils::Symbol * ParquetInputAdapterManager::getCurSymbol()
{
    if( m_symbolColumn.empty() )
        return nullptr;

    if( !m_cachedSymbolDispatcher )
        return nullptr;

    switch( m_symbolType )
    {
        case CspType::Type::STRING:
        {
            auto & curSymbol = m_cachedSymbolDispatcher -> getCurValue<std::string>();
            CSP_TRUE_OR_THROW_RUNTIME( curSymbol.has_value(),
                "Parquet file row contains row with no value for symbol column " << m_symbolColumn );
            m_curSymbol = curSymbol.value();
            break;
        }
        case CspType::Type::INT64:
        {
            auto & curSymbol = m_cachedSymbolDispatcher -> getCurValue<int64_t>();
            CSP_TRUE_OR_THROW_RUNTIME( curSymbol.has_value(),
                "Parquet file row contains row with no value for symbol column " << m_symbolColumn );
            m_curSymbol = curSymbol.value();
            break;
        }
        default:
            CSP_THROW( RuntimeException, "Unexpected symbol type: " << m_symbolType );
    }
    return &m_curSymbol;
}

void ParquetInputAdapterManager::start( DateTime starttime, DateTime endtime )
{
    if( !m_startTime.isNone() )
    {
        starttime = std::max( starttime, m_startTime );
    }
    AdapterManager::start( starttime, endtime );
    CSP_TRUE_OR_THROW_RUNTIME( m_processor == nullptr, "Starting parquet adapter manager more than once" );

    // Collect needed columns for the main processor
    if( m_symbolColumn != "" )
        m_neededColumns.insert( m_symbolColumn );
    m_neededColumns.insert( m_timeColumn );

    for( auto && it : m_dictBasketInputAdapters )
        m_neededColumns.insert( it.first + "__csp_value_count" );

    // Build projection set so the stream source only reads columns we need
    std::set<std::string> projectionColumns = m_neededColumns;
    collectAdapterColumns( m_simInputAdapters, projectionColumns );

    for( auto && dictPair : m_dictBasketInputAdapters )
    {
        auto & basketName = dictPair.first;
        projectionColumns.insert( basketName + "__csp_symbol" );
        collectAdapterColumns( dictPair.second, projectionColumns );

        // Build column → basket ownership map from subscriptions
        m_columnToBasketName[ basketName + "__csp_symbol" ] = basketName;
        std::set<std::string> basketCols;
        collectAdapterColumns( dictPair.second, basketCols );
        for( auto & col : basketCols )
            m_columnToBasketName[ col ] = basketName;
    }

    m_streamSource -> init( starttime, endtime, projectionColumns );

    // Create processor and get first stream
    m_processor = std::make_unique<arrow::RecordBatchRowProcessor>();

    if( !m_streamSource -> nextStream() )
        return;  // no data

    m_curSchema = buildLogicalSchema( m_streamSource -> columnReaders() );
    m_hasData = true;

    setupProcessor( *m_processor, m_curSchema, m_neededColumns, m_simInputAdapters, true );

    // Determine symbol type
    if( !m_symbolColumn.empty() )
    {
        auto * symDispatcher = m_processor -> getDispatcher( m_symbolColumn );
        CSP_TRUE_OR_THROW_RUNTIME( symDispatcher != nullptr, "Symbol column '" << m_symbolColumn << "' not found" );
        detectSymbolType( symDispatcher );
    }

    subscribeAdapters( *m_processor, m_simInputAdapters, true, m_structSubscriptions );

    // Dict basket setup — must happen before bindSourcesFromReaders
    setupDictBaskets();

    bindSourcesFromReaders();

    // Validate time column
    auto * timeDispatcher = m_processor -> getDispatcher( m_timeColumn );
    CSP_TRUE_OR_THROW_RUNTIME( timeDispatcher != nullptr, "Time column '" << m_timeColumn << "' not found" );
    CSP_TRUE_OR_THROW_RUNTIME( timeDispatcher -> arrowTypeId() == ::arrow::Type::TIMESTAMP,
        "Time column must be timestamp type" );
    m_cachedTimeDispatcher = timeDispatcher;

    // Read first row
    if( !readNextRow() )
    {
        m_hasData = false;
        return;
    }
}

void ParquetInputAdapterManager::stop()
{
    m_processor.reset();
    m_cachedTimeDispatcher = nullptr;
    m_cachedSymbolDispatcher = nullptr;
    m_curSchema.reset();
    m_mainRBSources.clear();
    m_dictBasketReaders.clear();
    m_structSubscriptions.clear();
    m_hasData = false;
    m_streamSource = nullptr;
    AdapterManager::stop();
}

void ParquetInputAdapterManager::processDictBaskets( bool dispatch )
{
    for( auto && record : m_dictBasketReaders )
    {
        auto numValues = record.getValueCount();
        const char * phase = dispatch ? "dispatch" : "skip";
        for( uint16_t i = 0; i < numValues; ++i )
        {
            if( dispatch )
            {
                CSP_TRUE_OR_THROW_RUNTIME( record.m_processor -> readRowAndAdvance(),
                    "Dict basket '" << record.m_basketName
                    << "' exhausted during " << phase << ": expected " << numValues
                    << " rows but only had " << i );
                const utils::Symbol * basketSymbol = nullptr;
                utils::Symbol tmpSymbol;
                if( record.m_cachedSymbolDispatcher &&
                    extractSymbolFromDispatcher( record.m_cachedSymbolDispatcher, tmpSymbol ) )
                    basketSymbol = &tmpSymbol;
                record.m_processor -> dispatchRow( basketSymbol );
                for( auto & structSub : record.m_structSubscriptions )
                    structSub -> dispatchValue( basketSymbol );
            }
            else
            {
                CSP_TRUE_OR_THROW_RUNTIME( record.m_processor -> skipRow(),
                    "Dict basket '" << record.m_basketName
                    << "' exhausted during " << phase << ": expected " << numValues
                    << " rows but only had " << i );
            }
        }
    }
}

DateTime ParquetInputAdapterManager::processNextSimTimeSlice( DateTime time )
{
    if( !m_hasData ) [[unlikely]]
    {
        return DateTime::NONE();
    }

    auto data_reference_time = time - m_time_shift;
    auto nextDataTime = m_cachedTimeDispatcher -> getCurValue<DateTime>();
    CSP_TRUE_OR_THROW_RUNTIME( nextDataTime.has_value(), "Null value in time column '" << m_timeColumn << "'" );

    while( !nextDataTime.value().isNone() && nextDataTime.value() < data_reference_time )
    {
        processDictBaskets( false );
        if( !readNextRow() )
        {
            nextDataTime = DateTime::NONE();
            break;
        }
        nextDataTime = m_cachedTimeDispatcher -> getCurValue<DateTime>();
        CSP_TRUE_OR_THROW_RUNTIME( nextDataTime.has_value(), "Null value in time column '" << m_timeColumn << "'" );
    }

    if( nextDataTime.value().isNone() || ( !m_endTime.isNone() && ( m_endTime - m_time_shift ) < nextDataTime.value() ) ) [[unlikely]]
    {
        return DateTime::NONE();
    }

    if( nextDataTime.value() > data_reference_time )
    {
        return nextDataTime.value() + m_time_shift;
    }

    CSP_TRUE_OR_THROW_RUNTIME( data_reference_time == nextDataTime.value(), "Expected time " << nextDataTime.value() << " got " << data_reference_time );
    do
    {
        processDictBaskets( true );

        auto * symbol = getCurSymbol();
        m_processor -> dispatchRow( symbol );
        for( auto & structSub : m_structSubscriptions )
            structSub -> dispatchValue( symbol );

        if( !readNextRow() )
        {
            nextDataTime = DateTime::NONE();
            break;
        }
        nextDataTime = m_cachedTimeDispatcher -> getCurValue<DateTime>();
        CSP_TRUE_OR_THROW_RUNTIME( nextDataTime.has_value(), "Null value in time column '" << m_timeColumn << "'" );
    } while( !nextDataTime.value().isNone() && nextDataTime.value() == data_reference_time );

    if( nextDataTime.value().isNone() ) [[unlikely]]
    {
        return DateTime::NONE();
    }

    if( m_allowOverlappingPeriods && nextDataTime.value() < data_reference_time )
    {
        return time + TimeDelta( 0, 1 );
    }
    return nextDataTime.value() + m_time_shift;
}


ManagedSimInputAdapter *ParquetInputAdapterManager::getInputAdapter( CspTypePtr &type, const Dictionary &properties, PushMode pushMode )
{
    CSP_TRUE_OR_THROW( !m_pushMode.has_value() || m_pushMode.value() == pushMode, NotImplemented,
                       "Subscribing with varying push modes is not currently supported. previous=" << m_pushMode.value()
                                                                                                   << " current=" << pushMode );
    m_pushMode = pushMode;
    std::string basketName = properties.get<std::string>( "basket_name", "" );

    utils::Symbol symbol = "";
    if( properties.exists( "symbol" ) )
    {
        auto rawSymbol = properties.getUntypedValue( "symbol" );
        if( std::holds_alternative<std::string>( rawSymbol ) )
            symbol = std::get<std::string>( rawSymbol );
        else if( std::holds_alternative<int64_t>( rawSymbol ) )
            symbol = std::get<int64_t>( rawSymbol );
        else
            CSP_THROW( TypeError, "Parquet subscribe symbol must be a string or int type" );
    }

    if( basketName.empty() )
    {
        return getRegularAdapter( type, properties, pushMode, symbol );
    }
    else
    {
        CSP_TRUE_OR_THROW( type -> type() != CspType::Type::DIALECT_GENERIC, NotImplemented, "Reading of baskets of arrays is unsupported" );
        return getDictBasketAdapter( type, properties, pushMode, symbol, basketName );
    }
}

ManagedSimInputAdapter *
ParquetInputAdapterManager::getRegularAdapter( const CspTypePtr &type, const Dictionary &properties, const PushMode &pushMode,
                                               const utils::Symbol &symbol )
{
    if( pushMode == PushMode::NON_COLLAPSING )
    {
        if( std::holds_alternative<std::string>( symbol) && std::get<std::string>( symbol ).empty() )
        {
            m_subscribedForAll = true;
        }
        else
        {
            m_subscribedBySymbol = true;
        }
        // Mixing by-symbol and subscribe-all causes rows for the same timestamp
        // to tick on different engine cycles, so disallow it.
        CSP_TRUE_OR_THROW( !m_subscribedBySymbol || !m_subscribedForAll, NotImplemented,
                           "Subscribing both by symbol and without symbol for same parquet reader is not currently supported" );
    }

    auto fieldMap = properties.getUntypedValue( "field_map" );
    if( std::holds_alternative<std::string>( fieldMap ) )
    {
        auto field = properties.get<std::string>( "field_map" );
        return getOrCreateSingleColumnAdapter( m_simInputAdapters, type, symbol, field, pushMode );
    }
    else if( std::holds_alternative<DictionaryPtr>( fieldMap ) )
    {
        CSP_TRUE_OR_THROW( type -> type() != CspType::Type::DIALECT_GENERIC, NotImplemented, "Reading of arrays of structs is unsupported" );
        auto dictFieldMap = properties.get<DictionaryPtr>( "field_map" );
        return getOrCreateStructColumnAdapter( m_simInputAdapters, type, symbol, dictFieldMap, pushMode );
    }
    properties.get<std::string>( "field_map" );
    CSP_THROW( RuntimeException, "Unexpected field_map type" );
}

ManagedSimInputAdapter *ParquetInputAdapterManager::getDictBasketAdapter( const CspTypePtr &type,
                                                                          const Dictionary &properties, const PushMode &pushMode,
                                                                          const utils::Symbol &symbol,
                                                                          const std::string &basketName )
{
    auto fieldMap = properties.getUntypedValue( "field_map" );

    auto &&basketRecord = m_dictBasketInputAdapters[ basketName ];

    if( std::holds_alternative<std::string>( fieldMap ) )
    {
        auto field = properties.get<std::string>( "field_map" );
        CSP_TRUE_OR_THROW_RUNTIME( field.empty(), "Non empty field map for dict basket" );
        return getOrCreateSingleColumnAdapter( basketRecord, type, symbol, basketName, pushMode );
    }
    else if( std::holds_alternative<DictionaryPtr>( fieldMap ) )
    {
        auto fieldMapTyped = properties.get<DictionaryPtr>( "field_map" );
        return getOrCreateStructColumnAdapter( basketRecord, type, symbol, fieldMapTyped, pushMode );
    }
    properties.get<std::string>( "field_map" );
    CSP_THROW( RuntimeException, "Unexpected field_map type" );
}

ManagedSimInputAdapter *
ParquetInputAdapterManager::getOrCreateSingleColumnAdapter( ParquetInputAdapterManager::AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol, const std::string &field,
                                                            const PushMode &pushMode )
{
    auto [itBySymbol, _inserted1] = inputAdaptersContainer.try_emplace( symbol );

    auto itByColumn = itBySymbol -> second.m_adaptersByColumnName.find( field );

    if( itByColumn == itBySymbol -> second.m_adaptersByColumnName.end() )
    {
        itByColumn = itBySymbol -> second.m_adaptersByColumnName.emplace(
                field, AdapterInfo{ engine() -> createOwnedObject<ManagedSimInputAdapter>(
                        type, this,
                        pushMode ) } ).first;
    }
    else
    {
        // Validate that the requested type matches the existing adapter
        CSP_TRUE_OR_THROW( itByColumn -> second.m_adapter -> dataType() -> type() == type -> type(), TypeError,
            "Conflicting types for column '" << field << "': existing adapter has type "
            << itByColumn -> second.m_adapter -> dataType() -> type().asCString()
            << " but new subscription requests " << type -> type().asCString() );
    }
    return itByColumn -> second.m_adapter;
}



ManagedSimInputAdapter *ParquetInputAdapterManager::getOrCreateStructColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                                                    const CspTypePtr &type, const utils::Symbol &symbol,
                                                                                    const csp::DictionaryPtr &fieldMap,
                                                                                    const PushMode &pushMode )
{
    auto [itBySymbol, _inserted2] = inputAdaptersContainer.try_emplace( symbol );

    StructAdapterInfo key{ type, fieldMap };

    auto itByColumn = itBySymbol -> second.m_structAdapters.find( key );

    if( itByColumn == itBySymbol -> second.m_structAdapters.end() )
    {
        itByColumn = itBySymbol -> second.m_structAdapters.emplace(
                key, engine() -> createOwnedObject<ManagedSimInputAdapter>(
                        type, this,
                        pushMode ) ).first;
    }
    return itByColumn -> second.m_adapter;

}


}
