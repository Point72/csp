#include <csp/adapters/parquet/ParquetInputAdapterManager.h>
#include <csp/adapters/arrow/ArrowTypeVisitor.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/TypeCast.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <arrow/type_traits.h>

namespace csp::adapters::parquet
{

namespace
{

// Create a field setter lambda that reads from a ColumnDispatcher and writes to a Struct field.
using FieldSetter = std::function<void( StructPtr & )>;

template<typename ColType>
FieldSetter makeFieldSetter( const StructFieldPtr & fieldPtr, arrow::ColumnDispatcher & dispatcher )
{
    using TypeSwitch = ConstructibleTypeSwitch<ColType>;
    arrow::ColumnDispatcher * dispPtr = &dispatcher;
    return TypeSwitch::invoke( fieldPtr -> type().get(),
        [dispPtr, fieldPtr]( auto tag )
        {
            using FieldType = typename decltype(tag)::type;
            return FieldSetter( [dispPtr, fieldPtr]( StructPtr & s )
            {
                auto & val = dispPtr -> getCurValue<ColType>();
                if( val.has_value() )
                    fieldPtr -> setValue<FieldType>( s.get(), csp::cast<FieldType>( val.value() ) );
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
        auto enumMetaPtr = std::static_pointer_cast<const CspEnumType>( fieldPtr -> type() ) -> meta();
        return FieldSetter( [dispPtr, fieldPtr, enumMetaPtr]( StructPtr & s )
        {
            auto & val = dispPtr -> getCurValue<std::string>();
            if( val.has_value() )
                fieldPtr -> setValue<CspEnum>( s.get(), enumMetaPtr -> fromString( val.value().c_str() ) );
        } );
    }

    // Map arrow type to C++ column type via visitor
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

} // anonymous namespace

// Helper to validate arrow type is compatible with subscriber's CspType
static void ensureType( const std::string & columnName, ::arrow::Type::type arrowType, const std::string & arrowTypeName,
                        const CspType * cspType )
{
    using T = CspType::Type;
    auto t = cspType -> type();

    bool compatible = false;
    switch( arrowType )
    {
        case ::arrow::Type::BOOL:
            compatible = ( t == T::BOOL );
            break;
        case ::arrow::Type::INT8:
        case ::arrow::Type::INT16:
        case ::arrow::Type::INT32:
        case ::arrow::Type::INT64:
        case ::arrow::Type::UINT8:
        case ::arrow::Type::UINT16:
        case ::arrow::Type::UINT32:
        case ::arrow::Type::UINT64:
            compatible = ( t == T::INT8 || t == T::UINT8 || t == T::INT16 || t == T::UINT16 ||
                           t == T::INT32 || t == T::UINT32 || t == T::INT64 || t == T::UINT64 ||
                           t == T::DOUBLE );
            break;
        case ::arrow::Type::HALF_FLOAT:
        case ::arrow::Type::FLOAT:
        case ::arrow::Type::DOUBLE:
            compatible = ( t == T::DOUBLE );
            break;
        case ::arrow::Type::STRING:
        case ::arrow::Type::LARGE_STRING:
        case ::arrow::Type::BINARY:
        case ::arrow::Type::LARGE_BINARY:
        case ::arrow::Type::FIXED_SIZE_BINARY:
        case ::arrow::Type::DICTIONARY:
            compatible = ( t == T::STRING || t == T::ENUM );
            break;
        case ::arrow::Type::TIMESTAMP:
            compatible = ( t == T::DATETIME );
            break;
        case ::arrow::Type::DURATION:
            compatible = ( t == T::TIMEDELTA );
            break;
        case ::arrow::Type::DATE32:
        case ::arrow::Type::DATE64:
            compatible = ( t == T::DATE );
            break;
        case ::arrow::Type::TIME32:
        case ::arrow::Type::TIME64:
            compatible = ( t == T::TIME );
            break;
        case ::arrow::Type::LIST:
        case ::arrow::Type::LARGE_LIST:
            compatible = ( t == T::DIALECT_GENERIC );
            break;
        case ::arrow::Type::STRUCT:
            compatible = ( t == T::STRUCT );
            break;
        default:
            compatible = false;
    }

    CSP_TRUE_OR_THROW( compatible, TypeError,
        "Unexpected column type for column " << columnName << " , expected " << t.asCString()
        << " got " << arrowTypeName );
}

// --- StructSubscription ---

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
            continue;  // missing column (allowMissing)

        auto fieldPtr = m_structMeta -> field( structFieldName );
        CSP_TRUE_OR_THROW_RUNTIME( fieldPtr != nullptr,
            "No field '" << structFieldName << "' in struct " << m_structMeta -> name() );

        // For STRUCT-type fields, ensure the dispatcher exists with proper StructMeta.
        // If it doesn't already exist (because setupFromSchema skipped it), create and add it.
        if( fieldPtr -> type() -> type() == CspType::Type::STRUCT &&
            dispatcher -> arrowTypeId() != ::arrow::Type::STRUCT )
        {
            continue;  // type mismatch, skip
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

// --- ParquetInputAdapterManager core ---

ParquetInputAdapterManager::ParquetInputAdapterManager( csp::Engine *engine, const Dictionary &properties,
                                                        RecordBatchGeneratorPtr rbGeneratorPtr) :
        AdapterManager( engine ),
        m_time_shift(0, 0),
        m_rbGenerator(rbGeneratorPtr)
{
    CSP_TRUE_OR_THROW_RUNTIME( m_rbGenerator, "RecordBatch generator must be provided" );

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

ParquetInputAdapterManager::~ParquetInputAdapterManager()
{
}

bool ParquetInputAdapterManager::getNextBatch()
{
    RecordBatchWithFlag item;
    if( !m_rbGenerator -> next( item ) )
    {
        m_hasData  = false;
        m_curBatch = nullptr;
        m_curBasketBatches.clear();
        return false;
    }
    CSP_TRUE_OR_THROW_RUNTIME( item.batch, "RecordBatch generator yielded null batch" );

    auto schema = item.batch -> schema();
    m_schemaChanged = m_curSchema && !m_curSchema -> Equals( *schema );

    m_curBatch  = item.batch;
    m_curSchema = schema;
    m_curBasketBatches = std::move( item.basketBatches );
    m_hasData   = true;
    return true;
}

void ParquetInputAdapterManager::setupProcessor(
    arrow::RecordBatchRowProcessor & processor,
    const std::shared_ptr<::arrow::Schema> & schema,
    const std::set<std::string> & neededColumns,
    const AdaptersBySymbol & adaptersBySymbol,
    bool subscribeAllOnEmptySymbol )
{
    // Collect all needed columns: explicit + from adapters + from struct field maps.
    // Also collect StructMeta for any STRUCT columns so setupFromSchema can create
    // all dispatchers in a single pass.
    std::set<std::string> allColumns{ neededColumns };
    std::unordered_map<std::string, std::shared_ptr<const StructMeta>> structMetaByColumn;

    for( auto && adaptersForSymbol : adaptersBySymbol )
    {
        // Scalar column subscriptions
        for( auto && entry : adaptersForSymbol.second.m_adaptersByColumnName )
        {
            auto & colName = entry.first;
            allColumns.insert( colName );

            // If subscribing to a STRUCT column, extract meta from adapter type
            auto * adapterType = entry.second.m_adapter -> dataType();
            if( adapterType -> type() == CspType::Type::STRUCT )
            {
                auto meta = static_cast<const CspStructType *>( adapterType ) -> meta();
                auto [it2, inserted] = structMetaByColumn.emplace( colName, meta );
                if( !inserted && it2 -> second != meta )
                    CSP_THROW( TypeError, "Conflicting struct types for column '" << colName
                        << "': multiple subscriptions use different struct types for the same STRUCT column" );
            }
        }

        // Struct subscriptions with field maps
        for( auto && entry : adaptersForSymbol.second.m_structAdapters )
        {
            auto parentMeta = std::static_pointer_cast<const CspStructType>( entry.first.type() ) -> meta();
            auto && fieldMap = entry.first.fieldMap();
            for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
            {
                auto & colName = it.key();
                allColumns.insert( colName );

                // If the struct field is itself a STRUCT, extract its nested meta
                auto & structFieldName = it.value<std::string>();
                auto fieldPtr = parentMeta -> field( structFieldName );
                if( fieldPtr && fieldPtr -> type() -> type() == CspType::Type::STRUCT )
                {
                    auto meta = std::static_pointer_cast<const CspStructType>( fieldPtr -> type() ) -> meta();
                    auto [it2, inserted] = structMetaByColumn.emplace( colName, meta );
                    if( !inserted && it2 -> second != meta )
                        CSP_THROW( TypeError, "Conflicting struct types for column '" << colName
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
    bool subscribeAllOnEmptySymbol )
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

        // Scalar/list column subscriptions
        for( auto & columnPair : symMapPair.second.m_adaptersByColumnName )
        {
            if( !processor.hasColumn( columnPair.first ) )
                continue;  // missing column

            // List type validation: adapter type must match arrow column type
            auto * dispatcher = processor.getDispatcher( columnPair.first );
            bool isList = ( dispatcher -> arrowTypeId() == ::arrow::Type::LIST ||
                            dispatcher -> arrowTypeId() == ::arrow::Type::LARGE_LIST );
            bool isListAdapter = ( columnPair.second.m_adapter -> dataType() -> type() == CspType::Type::DIALECT_GENERIC );

            CSP_TRUE_OR_THROW_RUNTIME( !isList || isListAdapter,
                "Column " << columnPair.first << " is a list column while subscribing as non-list" );
            CSP_TRUE_OR_THROW_RUNTIME( isList || !isListAdapter,
                "Column " << columnPair.first << " is not a list column while subscribing as list" );

            // Type validation: check arrow column type matches subscriber's CspType
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

            // Find or create the StructSubscription
            StructSubscription * sub = nullptr;
            for( auto & existing : m_structSubscriptions )
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
                m_structSubscriptions.push_back( std::move( newSub ) );
            }

            sub -> addSubscriber( structPair.second.m_adapter, symbol );
            sub -> createFieldSetters( processor, m_curSchema );
        }
    }
}

bool ParquetInputAdapterManager::readNextRow()
{
    if( !m_hasData ) [[unlikely]]
        return false;

    // If processor has more rows, read from it
    if( m_processor -> hasMoreRows() )
    {
        m_processor -> readNextRow();
        return true;
    }

    // Current batch exhausted — get next non-empty batch
    do
    {
        if( !getNextBatch() )
            return false;
    } while( m_curBatch -> num_rows() == 0 );

    // Handle schema change (new file with different columns)
    if( m_schemaChanged )
    {
        m_structSubscriptions.clear();
        setupProcessor( *m_processor, m_curSchema, m_neededColumns, m_simInputAdapters, true );
        subscribeAdapters( *m_processor, m_simInputAdapters, true );

        // C3: Rebuild dict basket readers after schema change.
        // setupProcessor destroys and recreates dispatchers in m_processor,
        // invalidating the raw m_valueCountDispatcher pointers.  Also rebuild
        // basket processors since basket batch schemas may have changed.
        for( auto & record : m_dictBasketReaders )
        {
            auto adapterIt = m_dictBasketInputAdapters.find( record.m_basketName );
            if( adapterIt == m_dictBasketInputAdapters.end() )
                continue;

            auto vcName = record.m_basketName + "__csp_value_count";
            record.m_valueCountDispatcher = m_processor -> getDispatcher( vcName );

            std::string symCol = record.m_basketName + "__csp_symbol";
            std::set<std::string> basketColumns{ symCol };
            std::unordered_map<std::string, std::shared_ptr<const StructMeta>> basketStructMeta;
            for( auto && symPair : adapterIt -> second )
            {
                for( auto && colPair : symPair.second.m_adaptersByColumnName )
                    basketColumns.insert( colPair.first );

                for( auto && structPair : symPair.second.m_structAdapters )
                {
                    auto parentMeta = std::static_pointer_cast<const CspStructType>( structPair.first.type() ) -> meta();
                    auto && fieldMap = structPair.first.fieldMap();
                    for( auto fmIt = fieldMap -> begin(); fmIt != fieldMap -> end(); ++fmIt )
                    {
                        auto & colName = fmIt.key();
                        basketColumns.insert( colName );

                        auto & structFieldName = fmIt.value<std::string>();
                        auto fieldPtr = parentMeta -> field( structFieldName );
                        if( fieldPtr && fieldPtr -> type() -> type() == CspType::Type::STRUCT )
                        {
                            auto meta = std::static_pointer_cast<const CspStructType>( fieldPtr -> type() ) -> meta();
                            basketStructMeta.emplace( colName, meta );
                        }
                    }
                }
            }

            auto batchIt = m_curBasketBatches.find( record.m_basketName );
            if( batchIt != m_curBasketBatches.end() )
                record.m_processor -> setupFromSchema( batchIt -> second -> schema(), basketColumns, m_allowMissingColumns, basketStructMeta );
            else
                record.m_processor -> setupFromSchema( m_curSchema, basketColumns, m_allowMissingColumns, basketStructMeta );

            record.m_structSubscriptions.clear();
            auto basketSchema = ( batchIt != m_curBasketBatches.end() )
                ? batchIt -> second -> schema() : m_curSchema;
            for( auto & symMapPair : adapterIt -> second )
            {
                std::optional<utils::Symbol> symbol = symMapPair.first;
                for( auto & columnPair : symMapPair.second.m_adaptersByColumnName )
                {
                    if( record.m_processor -> hasColumn( columnPair.first ) )
                        record.m_processor -> addSubscriber( columnPair.first, columnPair.second.m_adapter, symbol );
                }
                for( auto & structPair : symMapPair.second.m_structAdapters )
                {
                    auto structMeta = std::static_pointer_cast<const CspStructType>( structPair.first.type() ) -> meta();
                    StructSubscription * sub = nullptr;
                    for( auto & existing : record.m_structSubscriptions )
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
                        record.m_structSubscriptions.push_back( std::move( newSub ) );
                    }
                    sub -> addSubscriber( structPair.second.m_adapter, symbol );
                    sub -> createFieldSetters( *record.m_processor, basketSchema );
                }
            }
        }

        m_schemaChanged = false;
    }

    m_processor -> bindBatch( *m_curBatch );
    // Rebind dict basket processors to their new basket batches
    for( auto && record : m_dictBasketReaders )
    {
        auto it = m_curBasketBatches.find( record.m_basketName );
        if( it != m_curBasketBatches.end() )
            record.m_processor -> bindBatch( *it -> second );
    }
    m_processor -> readNextRow();
    return true;
}

const utils::Symbol * ParquetInputAdapterManager::getCurSymbol()
{
    if( m_symbolColumn.empty() )
        return nullptr;

    auto * dispatcher = m_processor -> getDispatcher( m_symbolColumn );
    if( !dispatcher )
        return nullptr;

    switch( m_symbolType )
    {
        case CspType::Type::STRING:
        {
            auto & curSymbol = dispatcher -> getCurValue<std::string>();
            CSP_TRUE_OR_THROW_RUNTIME( curSymbol.has_value(),
                "Parquet file row contains row with no value for symbol column " << m_symbolColumn );
            m_curSymbol = curSymbol.value();
            break;
        }
        case CspType::Type::INT64:
        {
            auto & curSymbol = dispatcher -> getCurValue<int64_t>();
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
        starttime = std::max(starttime, m_startTime);
    }
    AdapterManager::start( starttime, endtime );
    CSP_TRUE_OR_THROW_RUNTIME( m_processor == nullptr, "Starting parquet adapter manager more than once" );

    m_rbGenerator->init(starttime, endtime);

    // Collect needed columns
    if( m_symbolColumn != "" )
        m_neededColumns.insert( m_symbolColumn );
    m_neededColumns.insert( m_timeColumn );

    for( auto && it : m_dictBasketInputAdapters )
        m_neededColumns.insert( it.first + "__csp_value_count" );

    // Create processor and get first batch
    m_processor = std::make_unique<arrow::RecordBatchRowProcessor>();

    if( !getNextBatch() )
        return;  // no data

    // Setup processor with schema and subscribe adapters
    setupProcessor( *m_processor, m_curSchema, m_neededColumns, m_simInputAdapters, true );

    // Determine symbol type BEFORE subscribing (validation needs it)
    if( !m_symbolColumn.empty() )
    {
        auto * symDispatcher = m_processor -> getDispatcher( m_symbolColumn );
        CSP_TRUE_OR_THROW_RUNTIME( symDispatcher != nullptr, "Symbol column '" << m_symbolColumn << "' not found" );
        auto symType = symDispatcher -> arrowTypeId();
        if( symType == ::arrow::Type::STRING || symType == ::arrow::Type::LARGE_STRING ||
            symType == ::arrow::Type::DICTIONARY )
            m_symbolType = CspType::Type::STRING;
        else if( symType == ::arrow::Type::INT64 )
            m_symbolType = CspType::Type::INT64;
        else
            CSP_THROW( TypeError, "Invalid symbol column type. Only string and int64 symbols are currently supported" );
    }

    subscribeAdapters( *m_processor, m_simInputAdapters, true );

    // Bind first batch to processor
    m_processor -> bindBatch( *m_curBatch );

    // Dict basket setup
    for( auto && it : m_dictBasketInputAdapters )
    {
        auto valueCountColumnName = it.first + "__csp_value_count";
        std::string basketSymbolColumn = it.first + "__csp_symbol";

        DictBasketReaderRecord record;
        record.m_basketName = it.first;
        record.m_valueCountDispatcher = m_processor -> getDispatcher( valueCountColumnName );
        CSP_TRUE_OR_THROW_RUNTIME( record.m_valueCountDispatcher != nullptr,
            "Value count column '" << valueCountColumnName << "' not found" );

        record.m_processor = std::make_unique<arrow::RecordBatchRowProcessor>();
        std::set<std::string> basketColumns{ basketSymbolColumn };
        std::unordered_map<std::string, std::shared_ptr<const StructMeta>> basketStructMeta;
        // Collect basket adapter columns (scalar + struct field columns)
        for( auto && symPair : it.second )
        {
            for( auto && colPair : symPair.second.m_adaptersByColumnName )
                basketColumns.insert( colPair.first );

            for( auto && structPair : symPair.second.m_structAdapters )
            {
                auto parentMeta = std::static_pointer_cast<const CspStructType>( structPair.first.type() ) -> meta();
                auto && fieldMap = structPair.first.fieldMap();
                for( auto fmIt = fieldMap -> begin(); fmIt != fieldMap -> end(); ++fmIt )
                {
                    auto & colName = fmIt.key();
                    basketColumns.insert( colName );

                    auto & structFieldName = fmIt.value<std::string>();
                    auto fieldPtr = parentMeta -> field( structFieldName );
                    if( fieldPtr && fieldPtr -> type() -> type() == CspType::Type::STRUCT )
                    {
                        auto meta = std::static_pointer_cast<const CspStructType>( fieldPtr -> type() ) -> meta();
                        basketStructMeta.emplace( colName, meta );
                    }
                }
            }
        }

        // Get basket batch schema for setupFromSchema
        auto basketBatchIt = m_curBasketBatches.find( it.first );
        if( basketBatchIt != m_curBasketBatches.end() )
        {
            record.m_processor -> setupFromSchema( basketBatchIt -> second -> schema(), basketColumns, m_allowMissingColumns, basketStructMeta );
        }
        else
        {
            record.m_processor -> setupFromSchema( m_curSchema, basketColumns, m_allowMissingColumns, basketStructMeta );
        }
        // Subscribe basket adapters (scalar)
        for( auto & symMapPair : it.second )
        {
            std::optional<utils::Symbol> symbol = symMapPair.first;
            for( auto & columnPair : symMapPair.second.m_adaptersByColumnName )
            {
                if( record.m_processor -> hasColumn( columnPair.first ) )
                    record.m_processor -> addSubscriber( columnPair.first, columnPair.second.m_adapter, symbol );
            }
        }
        // Subscribe basket adapters (struct)
        auto basketSchema = ( basketBatchIt != m_curBasketBatches.end() )
            ? basketBatchIt -> second -> schema() : m_curSchema;
        for( auto & symMapPair : it.second )
        {
            std::optional<utils::Symbol> symbol = symMapPair.first;
            for( auto & structPair : symMapPair.second.m_structAdapters )
            {
                auto structMeta = std::static_pointer_cast<const CspStructType>( structPair.first.type() ) -> meta();
                StructSubscription * sub = nullptr;
                for( auto & existing : record.m_structSubscriptions )
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
                    record.m_structSubscriptions.push_back( std::move( newSub ) );
                }
                sub -> addSubscriber( structPair.second.m_adapter, symbol );
                sub -> createFieldSetters( *record.m_processor, basketSchema );
            }
        }
        // Bind basket processor to its own basket batch
        if( basketBatchIt != m_curBasketBatches.end() )
            record.m_processor -> bindBatch( *basketBatchIt -> second );
        m_dictBasketReaders.push_back( std::move( record ) );
    }

    // Validate time column
    auto * timeDispatcher = m_processor -> getDispatcher( m_timeColumn );
    CSP_TRUE_OR_THROW_RUNTIME( timeDispatcher != nullptr, "Time column '" << m_timeColumn << "' not found" );
    CSP_TRUE_OR_THROW_RUNTIME( timeDispatcher -> arrowTypeId() == ::arrow::Type::TIMESTAMP,
        "Time column must be timestamp type" );

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
    m_curBatch.reset();
    m_curSchema.reset();
    m_dictBasketReaders.clear();
    m_structSubscriptions.clear();
    m_hasData = false;
    m_rbGenerator = nullptr;
    AdapterManager::stop();
}

DateTime ParquetInputAdapterManager::processNextSimTimeSlice( DateTime time )
{
    if( !m_hasData ) [[unlikely]]
    {
        return DateTime::NONE();
    }

    auto getTimeDispatcher = [this]() { return m_processor -> getDispatcher( m_timeColumn ); };
    auto data_reference_time = time - m_time_shift;
    auto nextDataTime = getTimeDispatcher() -> getCurValue<DateTime>();
    CSP_TRUE_OR_THROW_RUNTIME( nextDataTime.has_value(), "Null value in time column '" << m_timeColumn << "'" );

    while( !nextDataTime.value().isNone() && nextDataTime.value() < data_reference_time )
    {
        for( auto && dictBasketRecord : m_dictBasketReaders )
        {
            auto numValuesOpt = dictBasketRecord.m_valueCountDispatcher -> getCurValue<uint16_t>();
            CSP_TRUE_OR_THROW_RUNTIME( numValuesOpt.has_value(), "Null value in dict basket value count column" );
            auto numValuesToSkip = numValuesOpt.value();
            for( uint16_t i = 0; i < numValuesToSkip; ++i )
            {
                dictBasketRecord.m_processor -> skipRow();
            }
        }
        if( !readNextRow() )
        {
            nextDataTime = DateTime::NONE();
            break;
        }
        nextDataTime = getTimeDispatcher() -> getCurValue<DateTime>();
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
        // Dispatch dict baskets
        for( auto && dictBasketRecord : m_dictBasketReaders )
        {
            auto numValuesOpt = dictBasketRecord.m_valueCountDispatcher -> getCurValue<uint16_t>();
            CSP_TRUE_OR_THROW_RUNTIME( numValuesOpt.has_value(), "Null value in dict basket value count column" );
            auto numValuesToDispatch = numValuesOpt.value();
            std::string basketSymbolColumn = dictBasketRecord.m_basketName + "__csp_symbol";
            for( uint16_t i = 0; i < numValuesToDispatch; ++i )
            {
                dictBasketRecord.m_processor -> readNextRow();
                // Use the basket's own symbol column for routing
                auto * symDispatcher = dictBasketRecord.m_processor -> getDispatcher( basketSymbolColumn );
                const utils::Symbol * basketSymbol = nullptr;
                utils::Symbol tmpSymbol;
                if( symDispatcher )
                {
                    auto & symVal = symDispatcher -> getCurValue<std::string>();
                    if( symVal.has_value() )
                    {
                        tmpSymbol = symVal.value();
                        basketSymbol = &tmpSymbol;
                    }
                }
                dictBasketRecord.m_processor -> dispatchRow( basketSymbol );
                for( auto & structSub : dictBasketRecord.m_structSubscriptions )
                    structSub -> dispatchValue( basketSymbol );
            }
        }

        // Dispatch main row
        auto * symbol = getCurSymbol();
        m_processor -> dispatchRow( symbol );
        for( auto & structSub : m_structSubscriptions )
            structSub -> dispatchValue( symbol );

        // Read next row
        if( !readNextRow() )
        {
            nextDataTime = DateTime::NONE();
            break;
        }
        nextDataTime = getTimeDispatcher() -> getCurValue<DateTime>();
        CSP_TRUE_OR_THROW_RUNTIME( nextDataTime.has_value(), "Null value in time column '" << m_timeColumn << "'" );
    } while( !nextDataTime.value().isNone() && nextDataTime.value() == data_reference_time );

    if( nextDataTime.value().isNone() ) [[unlikely]]
    {
        return DateTime::NONE();
    }

    if(m_allowOverlappingPeriods && nextDataTime.value() < data_reference_time)
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
        // If we subscribe both by symbol and subscribe all, we might have weird issues. Consider the following scenario:
        // ts1: subscribe AAPL
        // ts2: subscribe IBM
        // ts3: subscribe_all
        // Assume that the file has 2 entries, both on the same timestamp, one for AAPL, one for IBM.
        // Then ts1 and ts2 will output on the first cycle the AAPL and IBM data, ts3 will output data for AAPL only, and on next
        // cycle will output the data for IBM. This behavior of same data ticking on different cycles could create a lot of issues
        // and generally buggy so we want to avoid it and forbid subscribing both ways.
        CSP_TRUE_OR_THROW( !m_subscribedBySymbol || !m_subscribedForAll, NotImplemented,
                           "Subscribing both by symbol and without symbol for same parquet reader is not currently supported" );
    }

    auto fieldMap = properties.getUntypedValue( "field_map" );
    if( std::holds_alternative<std::string>( fieldMap ) )
    {
        auto field = properties.get<std::string>( "field_map" );
        return getSingleColumnAdapter( type, symbol, field, pushMode );
    }
    else if( std::holds_alternative<DictionaryPtr>( fieldMap ) )
    {
        CSP_TRUE_OR_THROW( type -> type() != CspType::Type::DIALECT_GENERIC, NotImplemented, "Reading of arrays of structs is unsupported" );
        auto dictFieldMap = properties.get<DictionaryPtr>( "field_map" );
        return getStructAdapter( type, symbol, dictFieldMap, pushMode );
    }
    else
    {
        // throw exception
        properties.get<std::string>( "field_map" );
    }
    CSP_THROW( RuntimeException, "Reached unreachable code" );
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
    else
    {
        // throw exception
        properties.get<std::string>( "field_map" );
    }
    CSP_THROW( RuntimeException, "Reached unreachable code" );
}

ManagedSimInputAdapter *
ParquetInputAdapterManager::getOrCreateSingleColumnAdapter( ParquetInputAdapterManager::AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol, const std::string &field,
                                                            const PushMode &pushMode )
{
    auto itBySymbol = inputAdaptersContainer.find( symbol );
    if( itBySymbol == inputAdaptersContainer.end() )
    {
        itBySymbol = inputAdaptersContainer.emplace( symbol, AdaptersSingleSymbol() ).first;
    }

    auto itByColumn = itBySymbol -> second.m_adaptersByColumnName.find( field );

    if( itByColumn == itBySymbol -> second.m_adaptersByColumnName.end() )
    {
        itByColumn = itBySymbol -> second.m_adaptersByColumnName.emplace(
                field, AdapterInfo{ engine() -> createOwnedObject<ManagedSimInputAdapter>(
                        type, this,
                        pushMode ) } ).first;
    }
    return itByColumn -> second.m_adapter;
}

ManagedSimInputAdapter *
ParquetInputAdapterManager::getSingleColumnAdapter( const CspTypePtr &type, const utils::Symbol &symbol,
                                                    const std::string &field, PushMode pushMode )
{
    return getOrCreateSingleColumnAdapter( m_simInputAdapters, type, symbol, field, pushMode );
}


ManagedSimInputAdapter *ParquetInputAdapterManager::getOrCreateStructColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                                                    const CspTypePtr &type, const utils::Symbol &symbol,
                                                                                    const csp::DictionaryPtr &fieldMap,
                                                                                    const PushMode &pushMode )
{
    auto itBySymbol = inputAdaptersContainer.find( symbol );
    if( itBySymbol == inputAdaptersContainer.end() )
    {
        itBySymbol = inputAdaptersContainer.emplace( symbol, AdaptersSingleSymbol() ).first;
    }

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

ManagedSimInputAdapter *
ParquetInputAdapterManager::getStructAdapter( const CspTypePtr &type, const utils::Symbol &symbol,
                                              const csp::DictionaryPtr &fieldMap, PushMode pushMode )
{
    return getOrCreateStructColumnAdapter( m_simInputAdapters, type, symbol, fieldMap, pushMode );
}


}
