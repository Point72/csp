#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetInputAdapterManager_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetInputAdapterManager_H

#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <csp/adapters/arrow/RecordBatchRowProcessor.h>
#include <csp/adapters/parquet/RecordBatchWithFlag.h>
#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/adapters/utils/ValueDispatcher.h>
#include <csp/core/Generator.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <unordered_map>
#include <set>
#include <string>
#include <optional>


namespace csp::adapters::parquet
{


//Top level AdapterManager object for all parquet adapters in the engine
class ParquetInputAdapterManager final : public csp::AdapterManager
{
public:
    using RecordBatchGeneratorPtr = csp::Generator<RecordBatchWithFlag, csp::DateTime, csp::DateTime>::Ptr;

    ParquetInputAdapterManager( csp::Engine *engine, const Dictionary &properties,
                                RecordBatchGeneratorPtr rbGeneratorPtr );

    ~ParquetInputAdapterManager();

    const char *name() const override{ return "ParquetInputAdapterManager"; }

    //start subscriptions and processing thread 
    void start( DateTime starttime, DateTime endtime ) override;

    //stop subscriptions and processing thread
    void stop() override;

    DateTime processNextSimTimeSlice( DateTime time ) override;

    ManagedSimInputAdapter *getInputAdapter( CspTypePtr &type, const Dictionary &properties, PushMode pushMode );
private:
    using StructAdapterInfo = csp::adapters::utils::StructAdapterInfo;

    struct AdapterInfo
    {
        AdapterInfo(ManagedSimInputAdapter* adapter)
        : m_adapter(adapter)
        {
        }
        ManagedSimInputAdapter * m_adapter;
    };

    struct AdaptersSingleSymbol
    {
        using AdaptersByColumnName = std::unordered_map<std::string, AdapterInfo>;
        using StructAdaptersByStructKey = std::unordered_map<StructAdapterInfo, AdapterInfo>;
        AdaptersByColumnName      m_adaptersByColumnName;

        StructAdaptersByStructKey m_structAdapters;
    };

    // Tracks a struct subscription that combines multiple columns into a CSP Struct.
    struct StructSubscription
    {
        using FieldSetter      = std::function<void( StructPtr & )>;
        using ValueDispatcher  = csp::adapters::utils::ValueDispatcher<StructPtr &>;

        std::shared_ptr<const StructMeta> m_structMeta;
        DictionaryPtr                     m_fieldMap;
        std::vector<FieldSetter>          m_fieldSetters;
        ValueDispatcher                   m_valueDispatcher;
        bool                              m_needsReset = false;

        void createFieldSetters( arrow::RecordBatchRowProcessor & processor,
                                 const std::shared_ptr<::arrow::Schema> & schema );
        void dispatchValue( const utils::Symbol * symbol );
        void addSubscriber( ManagedSimInputAdapter * adapter, std::optional<utils::Symbol> symbol );
    };

    struct DictBasketReaderRecord
    {
        std::string                                     m_basketName;
        arrow::ColumnDispatcher *                        m_valueCountDispatcher = nullptr;
        std::unique_ptr<arrow::RecordBatchRowProcessor>  m_processor;
        std::vector<std::unique_ptr<StructSubscription>>  m_structSubscriptions;
    };

    using AdaptersBySymbol = std::unordered_map<utils::Symbol, AdaptersSingleSymbol>;
    using DictBasketSymbolAdapters = std::unordered_map<std::string, AdaptersBySymbol>;

    // Fetch the next record batch from the generator, handling schema changes.
    // Returns false when no more data is available.
    bool getNextBatch();

    // Set up a processor from the current batch schema and subscribe adapters.
    void setupProcessor( arrow::RecordBatchRowProcessor & processor,
                         const std::shared_ptr<::arrow::Schema> & schema,
                         const std::set<std::string> & neededColumns,
                         const AdaptersBySymbol & adaptersBySymbol,
                         bool subscribeAllOnEmptySymbol );

    // Subscribe adapters to a processor's dispatchers.
    void subscribeAdapters( arrow::RecordBatchRowProcessor & processor,
                            const AdaptersBySymbol & adaptersBySymbol,
                            bool subscribeAllOnEmptySymbol );

    // Read the next row into the main processor (handles batch boundaries).
    bool readNextRow();

    // Symbol handling
    const utils::Symbol * getCurSymbol();

    ManagedSimInputAdapter *getRegularAdapter( const CspTypePtr &type,
                                               const Dictionary &properties, const PushMode &pushMode, const utils::Symbol &symbol );
    ManagedSimInputAdapter *getDictBasketAdapter( const CspTypePtr &type,
                                                  const Dictionary &properties, const PushMode &pushMode, const utils::Symbol &symbol,
                                                  const std::string &basketName );

    ManagedSimInputAdapter *getOrCreateSingleColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol,
                                                            const std::string &field, const PushMode &pushMode );
    ManagedSimInputAdapter *getSingleColumnAdapter( const CspTypePtr &type,
                                                    const utils::Symbol &symbol, const std::string &field, PushMode pushMode );
    ManagedSimInputAdapter *getOrCreateStructColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol,
                                                            const csp::DictionaryPtr &fieldMap, const PushMode &pushMode );
    ManagedSimInputAdapter *getStructAdapter( const CspTypePtr &type, const utils::Symbol &symbol,
                                              const csp::DictionaryPtr &fieldMap, PushMode pushMode );


    DictBasketSymbolAdapters m_dictBasketInputAdapters;
    AdaptersBySymbol         m_simInputAdapters;


    csp::DateTime                       m_startTime;
    csp::DateTime                       m_endTime;
    csp::TimeDelta                      m_time_shift;
    RecordBatchGeneratorPtr             m_rbGenerator;
    std::string                         m_symbolColumn;
    std::string                         m_timeColumn;
    std::string                         m_defaultTimezone;
    bool                                m_allowOverlappingPeriods;
    bool                                m_allowMissingColumns;
    std::optional<PushMode>             m_pushMode;
    bool                                m_subscribedBySymbol      = false;
    bool                                m_subscribedForAll        = false;

    // Processor-based reader state
    std::unique_ptr<arrow::RecordBatchRowProcessor>  m_processor;
    std::shared_ptr<::arrow::RecordBatch>            m_curBatch;
    std::shared_ptr<::arrow::Schema>                 m_curSchema;
    std::unordered_map<std::string, std::shared_ptr<::arrow::RecordBatch>> m_curBasketBatches;
    std::set<std::string>                            m_neededColumns;
    bool                                             m_hasData = false;
    bool                                             m_schemaChanged = false;

    // Symbol column state
    CspType::Type                    m_symbolType;
    utils::Symbol                    m_curSymbol;

    // Struct subscriptions (assembled from multiple column dispatchers)
    std::vector<std::unique_ptr<StructSubscription>>  m_structSubscriptions;

    // Dict basket processors
    std::vector<DictBasketReaderRecord> m_dictBasketReaders;
};

}

#endif
