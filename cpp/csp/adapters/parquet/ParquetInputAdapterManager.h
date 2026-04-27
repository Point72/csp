#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetInputAdapterManager_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetInputAdapterManager_H

#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <csp/adapters/arrow/RecordBatchRowProcessor.h>
#include <csp/adapters/parquet/RecordBatchStreamSource.h>
#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/adapters/utils/ValueDispatcher.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <unordered_map>
#include <set>
#include <string>
#include <optional>


namespace csp::adapters::parquet
{


// Manages all parquet input adapters for a single engine run.
//
// Lifecycle:
//   1. Registration:  getInputAdapter() called per subscription (before engine starts)
//   2. start():       init stream source → create processors → wire adapters → read first row
//   3. processNextSimTimeSlice():  skip/dispatch loop per engine tick
//   4. stop():        tear down all state
class ParquetInputAdapterManager final : public csp::AdapterManager
{
public:
    using RecordBatchStreamSourcePtr = RecordBatchStreamSource::Ptr;

    ParquetInputAdapterManager( csp::Engine *engine, const Dictionary &properties,
                                RecordBatchStreamSourcePtr streamSource );

    ~ParquetInputAdapterManager();

    const char *name() const override{ return "ParquetInputAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;
    void stop() override;
    DateTime processNextSimTimeSlice( DateTime time ) override;

    ManagedSimInputAdapter *getInputAdapter( CspTypePtr &type, const Dictionary &properties, PushMode pushMode );

private:
    using StructAdapterInfo = csp::adapters::utils::StructAdapterInfo;

    // ── Adapter registration types ──────────────────────────────────

    struct AdapterInfo
    {
        AdapterInfo(ManagedSimInputAdapter* adapter) : m_adapter(adapter) {}
        ManagedSimInputAdapter * m_adapter;
    };

    struct AdaptersSingleSymbol
    {
        std::unordered_map<std::string, AdapterInfo>            m_adaptersByColumnName;
        std::unordered_map<StructAdapterInfo, AdapterInfo>      m_structAdapters;
    };

    // ── Struct subscriptions ─────────────────────────────────────────
    // Combines multiple column dispatchers into a single CSP Struct value.
    struct StructSubscription
    {
        using FieldSetter      = std::function<void( StructPtr & )>;
        using ValueDispatcher  = csp::adapters::utils::ValueDispatcher<StructPtr &>;

        std::shared_ptr<const StructMeta> m_structMeta;
        DictionaryPtr                     m_fieldMap;
        std::vector<FieldSetter>          m_fieldSetters;
        ValueDispatcher                   m_valueDispatcher;

        void createFieldSetters( arrow::RecordBatchRowProcessor & processor,
                                 const std::shared_ptr<::arrow::Schema> & schema );
        void dispatchValue( const utils::Symbol * symbol );
        void addSubscriber( ManagedSimInputAdapter * adapter, std::optional<utils::Symbol> symbol );
    };

    // ── Dict basket tracking ────────────────────────────────────────
    // A dict basket is a sub-table whose rows are keyed by a per-basket
    // symbol column and whose row count per main row is in a value-count column.
    struct DictBasketReaderRecord
    {
        std::string                                     m_basketName;
        std::string                                     m_basketSymbolColumn;
        arrow::ColumnDispatcher *                        m_valueCountDispatcher = nullptr;
        arrow::ColumnDispatcher *                        m_cachedSymbolDispatcher = nullptr;
        std::unique_ptr<arrow::RecordBatchRowProcessor>  m_processor;
        std::vector<std::unique_ptr<StructSubscription>>  m_structSubscriptions;

        std::vector<std::shared_ptr<::arrow::RecordBatchReader>> m_rbSources;

        uint16_t getValueCount() const;
    };

    using AdaptersBySymbol = std::unordered_map<utils::Symbol, AdaptersSingleSymbol>;
    using DictBasketSymbolAdapters = std::unordered_map<std::string, AdaptersBySymbol>;

    // ── Processor setup & wiring ─────────────────────────────────────

    static void collectAdapterColumns( const AdaptersBySymbol & adaptersBySymbol,
                                       std::set<std::string> & columns );

    void setupProcessor( arrow::RecordBatchRowProcessor & processor,
                         const std::shared_ptr<::arrow::Schema> & schema,
                         const std::set<std::string> & neededColumns,
                         const AdaptersBySymbol & adaptersBySymbol,
                         bool subscribeAllOnEmptySymbol );

    void subscribeAdapters( arrow::RecordBatchRowProcessor & processor,
                            const AdaptersBySymbol & adaptersBySymbol,
                            bool subscribeAllOnEmptySymbol,
                            std::vector<std::unique_ptr<StructSubscription>> & structSubscriptions );

    void setupBasketProcessor( DictBasketReaderRecord & record,
                               const AdaptersBySymbol & adaptersBySymbol );

    void setupDictBaskets();

    // ── Row reading & dispatch ───────────────────────────────────────

    bool readNextRow();
    void processDictBaskets( bool dispatch );
    bool advanceToNextStream();

    // ── Symbol handling ──────────────────────────────────────────────

    void detectSymbolType( arrow::ColumnDispatcher * symDispatcher );
    const utils::Symbol * getCurSymbol();

    // ── Source binding ───────────────────────────────────────────────

    static std::shared_ptr<::arrow::Schema> buildLogicalSchema( const ColumnReaderMap & readers );
    bool bindSourcesFromReaders();

    // ── Adapter registration ─────────────────────────────────────────

    ManagedSimInputAdapter *getRegularAdapter( const CspTypePtr &type,
                                               const Dictionary &properties, const PushMode &pushMode, const utils::Symbol &symbol );
    ManagedSimInputAdapter *getDictBasketAdapter( const CspTypePtr &type,
                                                  const Dictionary &properties, const PushMode &pushMode, const utils::Symbol &symbol,
                                                  const std::string &basketName );

    ManagedSimInputAdapter *getOrCreateSingleColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol,
                                                            const std::string &field, const PushMode &pushMode );
    ManagedSimInputAdapter *getOrCreateStructColumnAdapter( AdaptersBySymbol &inputAdaptersContainer,
                                                            const CspTypePtr &type, const utils::Symbol &symbol,
                                                            const csp::DictionaryPtr &fieldMap, const PushMode &pushMode );


    // ── Member state ────────────────────────────────────────────────

    // Registration-phase state (populated by getInputAdapter before start)
    DictBasketSymbolAdapters m_dictBasketInputAdapters;
    AdaptersBySymbol         m_simInputAdapters;

    // Configuration (from properties dict)
    csp::DateTime                       m_startTime;
    csp::DateTime                       m_endTime;
    csp::TimeDelta                      m_time_shift;
    RecordBatchStreamSourcePtr          m_streamSource;
    std::string                         m_symbolColumn;
    std::string                         m_timeColumn;
    std::string                         m_defaultTimezone;
    bool                                m_allowOverlappingPeriods;
    bool                                m_allowMissingColumns;
    std::optional<PushMode>             m_pushMode;
    bool                                m_subscribedBySymbol      = false;
    bool                                m_subscribedForAll        = false;

    // Runtime state (initialized in start, used in processNextSimTimeSlice)
    std::unique_ptr<arrow::RecordBatchRowProcessor>  m_processor;
    std::shared_ptr<::arrow::Schema>                 m_curSchema;
    std::set<std::string>                            m_neededColumns;
    bool                                             m_hasData = false;
    std::vector<std::shared_ptr<::arrow::RecordBatchReader>>    m_mainRBSources;
    CspType::Type                                    m_symbolType;
    utils::Symbol                                    m_curSymbol;
    std::vector<std::unique_ptr<StructSubscription>> m_structSubscriptions;
    std::vector<DictBasketReaderRecord>              m_dictBasketReaders;

    // Cached dispatchers (avoid repeated map lookups on the hot path)
    arrow::ColumnDispatcher *                m_cachedTimeDispatcher   = nullptr;
    arrow::ColumnDispatcher *                m_cachedSymbolDispatcher = nullptr;

    // Maps column name → basket name (built once in start)
    std::unordered_map<std::string, std::string> m_columnToBasketName;
};

}

#endif // _IN_CSP_ADAPTERS_PARQUET_ParquetInputAdapterManager_H
