#include <csp/adapters/parquet/ParquetInputAdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/adapters/parquet/ParquetReader.h>
#include <csp/adapters/parquet/ParquetReaderColumnAdapter.h>
#include <arrow/type_traits.h>

namespace csp::adapters::parquet
{

ParquetInputAdapterManager::ParquetInputAdapterManager( csp::Engine *engine, const Dictionary &properties,
                                                        GeneratorPtr generatorPtr,
                                                        TableGeneratorPtr tableGeneratorPtr) :
        AdapterManager( engine ),
        m_fileNameGeneratorReplicator( generatorPtr ? std::make_shared<FileNameGeneratorReplicator>( generatorPtr ) : nullptr ),
        m_time_shift(0, 0),
        m_tableGenerator(tableGeneratorPtr),
        m_reader()
{
    CSP_TRUE_OR_THROW_RUNTIME(!generatorPtr || !m_tableGenerator, "Trying to set both generatorPtr and tableGeneratorPtr");
    CSP_TRUE_OR_THROW_RUNTIME(generatorPtr || m_tableGenerator, "Either generatorPtr or tableGeneratorPtr must be set");

    m_symbolColumn        = properties.get<std::string>( "symbol_column", "" );
    m_timeColumn          = properties.get<std::string>( "time_column", "" );
    m_defaultTimezone     = properties.get<std::string>( "tz", "UTC" );
    m_splitColumnsToFiles = properties.get<bool>( "split_columns_to_files" );
    m_isArrowIPC = properties.get<bool>( "is_arrow_ipc", false );
    m_allowOverlappingPeriods = properties.get<bool>( "allow_overlapping_periods", false );
    m_allowMissingColumns = properties.get<bool>( "allow_missing_columns", false );
    m_allowMissingFiles = properties.get<bool>( "allow_missing_files", false );
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

void ParquetInputAdapterManager::start( DateTime starttime, DateTime endtime )
{
    if( !m_startTime.isNone() )
    {
        starttime = std::max(starttime, m_startTime);
    }
    AdapterManager::start( starttime, endtime );
    CSP_TRUE_OR_THROW_RUNTIME( m_reader == nullptr, "Starting parquet adapter manager more than once" );
    if(m_fileNameGeneratorReplicator)
    {
        m_fileNameGeneratorReplicator -> init( starttime, endtime );
    }
    else
    {
        m_tableGenerator->init(starttime, endtime);
    }

    std::optional<std::string> symbolColumn;
    std::set<std::string>      neededColumns;

    if( m_symbolColumn != "" )
    {
        neededColumns.insert( m_symbolColumn );
        symbolColumn = m_symbolColumn;
    }
    neededColumns.insert( m_timeColumn );

    for( auto &&it : m_dictBasketInputAdapters )
    {
        neededColumns.insert( it.first + "__csp_value_count" );
    }
    m_reader = initializeParquetReader( symbolColumn, neededColumns, m_simInputAdapters, true, true );
    if(m_reader == nullptr)
    {
        return;
    }

    for( auto &&it : m_dictBasketInputAdapters )
    {
        auto valueCountColumnName = it.first + "__csp_value_count";
        std::string basketSymbolColumn = it.first + +"__csp_symbol";
        DictBasketReaderRecord record{ ( *m_reader )[ valueCountColumnName ], nullptr };
        record.m_valueCountColumn -> ensureType( CspType::UINT16() );
        neededColumns.insert( valueCountColumnName );

        record.m_reader = initializeParquetReader( basketSymbolColumn, { basketSymbolColumn }, it.second,
                                                     false );
        m_dictBasketReaders.push_back( std::move( record ) );
    }

    m_timestampColumnAdapter = ( *m_reader )[ m_timeColumn ];
    CSP_TRUE_OR_THROW_RUNTIME( m_timestampColumnAdapter.valid(), "m_timestampColumnAdapter is NULL" );
    m_timestampColumnAdapter -> ensureType( CspType::DATETIME() );

}

std::unique_ptr<ParquetReader> ParquetInputAdapterManager::initializeParquetReader( const std::optional<std::string> &symbolColumn,
                                                                                    const std::set<std::string> &neededColumns,
                                                                                    const ParquetInputAdapterManager::AdaptersBySymbol &adaptersBySymbol,
                                                                                    bool subscribeAllOnEmptySymbol,
                                                                                    bool nullOnEmpty) const
{
    std::set<std::string> neededColumnsCopy{ neededColumns };
    for( auto &&adaptersForSymbol : adaptersBySymbol )
    {
        for( auto &&columnAdapterEntryIt : adaptersForSymbol.second.m_adaptersByColumnName )
        {
            neededColumnsCopy.insert(columnAdapterEntryIt.first);
        }
        for( auto &&structAdapterEntryIt : adaptersForSymbol.second.m_structAdapters )
        {
            auto &&fieldMap = structAdapterEntryIt.first.fieldMap();
            for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
            {
                neededColumnsCopy.insert( it.key() );
            }
        }
    }

    std::vector<std::string>       columns( neededColumnsCopy.begin(), neededColumnsCopy.end() );
    std::unique_ptr<ParquetReader> reader;

    if( m_splitColumnsToFiles )
    {
        CSP_TRUE_OR_THROW_RUNTIME(m_fileNameGeneratorReplicator, "Trying to read split columns from file while reading in memory tables");
        reader.reset( new MultipleFileParquetReader( m_fileNameGeneratorReplicator, columns, m_isArrowIPC, m_allowMissingColumns, symbolColumn ) );
    }
    else
    {
        if(m_fileNameGeneratorReplicator)
        {
            reader.reset( new SingleFileParquetReader( m_fileNameGeneratorReplicator -> getGeneratorReplica(), columns, m_isArrowIPC,
                                                         m_allowMissingColumns, m_allowMissingFiles, symbolColumn ) );
        }
        else
        {
            reader.reset( new InMemoryTableParquetReader( m_tableGenerator, columns, m_allowMissingColumns, symbolColumn ) );
        }
    }
    if(!reader->isEmpty())
    {
        for( auto &symMapPair : adaptersBySymbol )
        {
            std::optional<utils::Symbol> symbol;
            if( !subscribeAllOnEmptySymbol ||
                !std::holds_alternative<std::string>( symMapPair.first ) ||
                ( std::holds_alternative<std::string>( symMapPair.first ) &&
                  std::get<std::string>( symMapPair.first ) != "" ) )
            {
                symbol = symMapPair.first;
            }

            for( auto &columnAdapterPair:symMapPair.second.m_adaptersByColumnName )
            {
                auto &&columnAdapter       = ( *reader )[ columnAdapterPair.first ];
                auto &&listReaderInterface = columnAdapterPair.second.m_listReaderInterface;
                CSP_TRUE_OR_THROW_RUNTIME( !columnAdapter->isListType() || listReaderInterface != nullptr,
                                           "Column " << columnAdapterPair.first << " is a list column in parquet file "
                                                     << reader -> getCurFileOrTableName() << " while subscribing as non list" );
                CSP_TRUE_OR_THROW_RUNTIME( columnAdapter->isListType() || listReaderInterface == nullptr,
                                           "Column " << columnAdapterPair.first << " is a list is non list in parquet file "
                                                     << reader -> getCurFileOrTableName() << " while subscribing to it as list" );

                if(columnAdapter->isListType())
                {
                    reader->addListSubscriber(columnAdapterPair.first, columnAdapterPair.second.m_adapter,
                            symbol, listReaderInterface);

                }
                else
                {
                    reader -> addSubscriber( columnAdapterPair.first, columnAdapterPair.second.m_adapter,
                                             symbol );
                }
            }
            for( auto &structAdapterPair:symMapPair.second.m_structAdapters )
            {
                CSP_TRUE_OR_THROW_RUNTIME( structAdapterPair.second.m_listReaderInterface == nullptr,
                                           "Struct adapter is not expected to have list reader interface set" );
                reader -> getStructAdapter( structAdapterPair.first ).addSubscriber(
                        structAdapterPair.second.m_adapter, symbol );
            }
        }
    }
    if(!reader->start() && nullOnEmpty)
    {
        return std::unique_ptr<ParquetReader>(nullptr);
    }
    return reader;
}

void ParquetInputAdapterManager::stop()
{
    m_reader.reset( nullptr );
    m_dictBasketReaders.clear();
    m_timestampColumnAdapter      = nullptr;
    m_fileNameGeneratorReplicator = nullptr;
    m_tableGenerator = nullptr;
    AdapterManager::stop();
}

DateTime ParquetInputAdapterManager::processNextSimTimeSlice( DateTime time )
{
    if( unlikely( !m_reader || !m_reader -> hasData() ) )
    {
        return DateTime::NONE();
    }
    auto data_reference_time = time - m_time_shift;
    auto nextDataTime = m_timestampColumnAdapter -> getCurValue<DateTime>();
    while( !nextDataTime.value().isNone() && nextDataTime.value() < data_reference_time )
    {
        for( auto &&dictBasketRecord:m_dictBasketReaders )
        {
            auto numValuesToSkip = dictBasketRecord.m_valueCountColumn -> getCurValue<uint16_t>().value();
            dictBasketRecord.m_reader -> skipRows( numValuesToSkip );
        }
        if(!m_reader -> skipRow())
        {
            nextDataTime = DateTime::NONE();
            break;
        }
        nextDataTime = m_timestampColumnAdapter -> getCurValue<DateTime>();
    }
    if( unlikely( nextDataTime.value().isNone() || ( !m_endTime.isNone() && ( m_endTime - m_time_shift ) < nextDataTime ) ) )
    {
        return DateTime::NONE();
    }

    if( nextDataTime.value() > data_reference_time )
    {
        return nextDataTime.value() + m_time_shift;
    }

    CSP_TRUE_OR_THROW_RUNTIME( data_reference_time == nextDataTime, "Expected time " << nextDataTime.value() << " got " << data_reference_time );
    do
    {
        for( auto &&dictBasketRecord:m_dictBasketReaders )
        {
            auto numValuesToDispatch = dictBasketRecord.m_valueCountColumn -> getCurValue<uint16_t>().value();

            for( uint16_t i = 0; i < numValuesToDispatch; ++i )
            {
                dictBasketRecord.m_reader -> dispatchRow();
            }
        }
        m_reader -> dispatchRow();

        nextDataTime = m_reader -> hasData() ? m_timestampColumnAdapter -> getCurValue<DateTime>() : DateTime::NONE();
    } while( !nextDataTime.value().isNone() && nextDataTime == data_reference_time );

    if( unlikely( nextDataTime -> isNone() ) )
    {
        return DateTime::NONE();
    }

    if(m_allowOverlappingPeriods && nextDataTime.value() < data_reference_time)
    {
        return time + TimeDelta( 0, 1 );
    }
    return nextDataTime.value() + m_time_shift;
}


ManagedSimInputAdapter *ParquetInputAdapterManager::getInputAdapter( CspTypePtr &type, const Dictionary &properties, PushMode pushMode,
                                                                     const DialectGenericListReaderInterface::Ptr &listReaderInterface )
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
        return getRegularAdapter( type, properties, pushMode, symbol, listReaderInterface );
    }
    else
    {
        CSP_TRUE_OR_THROW(listReaderInterface == nullptr, NotImplemented, "Reading of baskets of arrays is unsupported");
        return getDictBasketAdapter( type, properties, pushMode, symbol, basketName );
    }
}

ManagedSimInputAdapter *
ParquetInputAdapterManager::getRegularAdapter( const CspTypePtr &type, const Dictionary &properties, const PushMode &pushMode,
                                               const utils::Symbol &symbol,
                                               const DialectGenericListReaderInterface::Ptr &listReaderInterface )
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
        return getSingleColumnAdapter( type, symbol, field, pushMode, listReaderInterface );
    }
    else if( std::holds_alternative<DictionaryPtr>( fieldMap ) )
    {
        CSP_TRUE_OR_THROW(listReaderInterface == nullptr, NotImplemented, "Reading of arrays of structs is unsupported");
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
                                                            const PushMode &pushMode,
                                                            const DialectGenericListReaderInterface::Ptr &listReaderInterface )
{
    auto itBySymbol = inputAdaptersContainer.find( symbol );
    if( itBySymbol == inputAdaptersContainer.end() )
    {
        itBySymbol = inputAdaptersContainer.emplace( symbol, AdaptersSingleSymbol() ).first;
    }

    auto itByColumn = itBySymbol -> second.m_adaptersByColumnName.find( field );
    const CspTypePtr& adapterType = (listReaderInterface==nullptr) ? type : CspType::DIALECT_GENERIC();

    if( itByColumn == itBySymbol -> second.m_adaptersByColumnName.end() )
    {
        itByColumn = itBySymbol -> second.m_adaptersByColumnName.emplace(
                field, AdapterInfo{ engine() -> createOwnedObject<ManagedSimInputAdapter>(
                        adapterType, this,
                        pushMode ), listReaderInterface } ).first;
    }
    return itByColumn -> second.m_adapter;
}

ManagedSimInputAdapter *
ParquetInputAdapterManager::getSingleColumnAdapter( const CspTypePtr &type, const utils::Symbol &symbol,
                                                    const std::string &field, PushMode pushMode,
                                                    const DialectGenericListReaderInterface::Ptr &listReaderInterface)
{
    return getOrCreateSingleColumnAdapter( m_simInputAdapters, type, symbol, field,
                                           pushMode, listReaderInterface );
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
