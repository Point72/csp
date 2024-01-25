#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetInputAdapterManager_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetInputAdapterManager_H

#include <csp/adapters/parquet/ParquetReader.h>
#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/core/Generator.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <unordered_map>
#include <set>
#include <string>
#include <optional>
#include <csp/adapters/parquet/DialectGenericListReaderInterface.h>


namespace csp::adapters::parquet
{


//Top level AdapterManager object for all parquet adapters in the engine
class ParquetInputAdapterManager final : public csp::AdapterManager
{
public:
    using GeneratorPtr = csp::Generator<std::string, csp::DateTime, csp::DateTime>::Ptr;
    using TableGeneratorPtr = csp::Generator<std::shared_ptr<arrow::Table>, csp::DateTime, csp::DateTime>::Ptr;

    ParquetInputAdapterManager( csp::Engine *engine, const Dictionary &properties, GeneratorPtr generatorPtr, TableGeneratorPtr tableGeneratorPtr );

    ~ParquetInputAdapterManager();

    const char *name() const override{ return "ParquetInputAdapterManager"; }

    //start subscriptions and processing thread 
    void start( DateTime starttime, DateTime endtime ) override;

    //stop subscriptions and processing thread
    void stop() override;

    DateTime processNextSimTimeSlice( DateTime time ) override;

    ManagedSimInputAdapter *getInputAdapter( CspTypePtr &type, const Dictionary &properties, PushMode pushMode,
                                             const DialectGenericListReaderInterface::Ptr &listReaderInterface = nullptr );
private:
    using StructAdapterInfo = csp::adapters::utils::StructAdapterInfo;

    struct AdapterInfo
    {
        AdapterInfo(ManagedSimInputAdapter* adapter, const DialectGenericListReaderInterface::Ptr listReaderInterface = nullptr)
        : m_adapter(adapter), m_listReaderInterface(listReaderInterface)
        {
        }
        ManagedSimInputAdapter * m_adapter;
        DialectGenericListReaderInterface::Ptr m_listReaderInterface = nullptr;
    };

    struct AdaptersSingleSymbol
    {
        using AdaptersByColumnName = std::unordered_map<std::string, AdapterInfo>;
        using StructAdaptersByStructKey = std::unordered_map<StructAdapterInfo, AdapterInfo>;
        AdaptersByColumnName      m_adaptersByColumnName;

        StructAdaptersByStructKey m_structAdapters;
    };
    struct DictBasketReaderRecord
    {
        ColumnAdapterReference         m_valueCountColumn;
        std::unique_ptr<ParquetReader> m_reader;
    };

    using AdaptersBySymbol = std::unordered_map<utils::Symbol, AdaptersSingleSymbol>;
    using DictBasketSymbolAdapters = std::unordered_map<std::string, AdaptersBySymbol>;

    std::unique_ptr<ParquetReader> initializeParquetReader( const std::optional<std::string> &symbolColumn,
                                                            const std::set<std::string> &neededColumns,
                                                            const AdaptersBySymbol &adaptersBySymbol,
                                                            bool subscribeAllOnEmptySymbol = true,
                                                            bool nullOnEmpty = false ) const;

    ManagedSimInputAdapter *getRegularAdapter( const CspTypePtr &type,
                                               const Dictionary &properties, const PushMode &pushMode, const utils::Symbol &symbol,
                                               const DialectGenericListReaderInterface::Ptr &listReaderInterface = nullptr);
    ManagedSimInputAdapter *getDictBasketAdapter( const CspTypePtr &type,
                                                  const Dictionary &properties, const PushMode &pushMode, const utils::Symbol &symbol,
                                                  const std::string &basketName );

    ManagedSimInputAdapter *getOrCreateSingleColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol,
                                                            const std::string &field, const PushMode &pushMode,
                                                            const DialectGenericListReaderInterface::Ptr &listReaderInterface = nullptr );
    ManagedSimInputAdapter *getSingleColumnAdapter( const CspTypePtr &type,
                                                    const utils::Symbol &symbol, const std::string &field, PushMode pushMode,
                                                    const DialectGenericListReaderInterface::Ptr &listReaderInterface = nullptr);
    ManagedSimInputAdapter *getOrCreateStructColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol,
                                                            const csp::DictionaryPtr &fieldMap, const PushMode &pushMode );
    ManagedSimInputAdapter *getStructAdapter( const CspTypePtr &type, const utils::Symbol &symbol,
                                              const csp::DictionaryPtr &fieldMap, PushMode pushMode );


    DictBasketSymbolAdapters m_dictBasketInputAdapters;
    AdaptersBySymbol         m_simInputAdapters;


    FileNameGeneratorReplicator::Ptr    m_fileNameGeneratorReplicator;
    csp::DateTime                       m_startTime;
    csp::DateTime                       m_endTime;
    csp::TimeDelta                      m_time_shift;
    TableGeneratorPtr                   m_tableGenerator;
    std::string                         m_symbolColumn;
    std::string                         m_timeColumn;
    std::string                         m_defaultTimezone;
    bool                                m_splitColumnsToFiles;
    bool                                m_isArrowIPC;
    bool                                m_allowOverlappingPeriods;
    bool                                m_allowMissingColumns;
    bool                                m_allowMissingFiles;
    std::unique_ptr<ParquetReader>      m_reader;
    ColumnAdapterReference              m_timestampColumnAdapter;
    std::vector<DictBasketReaderRecord> m_dictBasketReaders;
    std::optional<PushMode>             m_pushMode;
    bool                                m_subscribedBySymbol      = false;
    bool                                m_subscribedForAll        = false;


};

}

#endif
