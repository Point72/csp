#ifndef _IN_CSP_ADAPTERS_CSV_CsvInputAdapterManager_H
#define _IN_CSP_ADAPTERS_CSV_CsvInputAdapterManager_H

#include "csp/core/Time.h"
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <fstream>
#include <functional>
#include <unordered_map>
#include <set>
#include <string>
#include <optional>
#include <variant>
#include <vector>


namespace csp::adapters::csv
{


// Manages all csv input adapters for a single engine run.
//
// Lifecycle:
//   1. Registration:  getInputAdapter() called per subscription (before engine starts)
//   2. start():       create processors → wire adapters → read first row
//   3. processNextSimTimeSlice():  skip/dispatch loop per engine tick
//   4. stop():        tear down all state
class CsvInputAdapterManager final : public csp::AdapterManager
{
public:

    CsvInputAdapterManager( csp::Engine *engine, const Dictionary &properties );

    ~CsvInputAdapterManager();

    const char *name() const override{ return "CsvInputAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;
    void stop() override;
    DateTime processNextSimTimeSlice( DateTime time ) override;

    ManagedSimInputAdapter * getInputAdapter( CspTypePtr & type, const Dictionary & properties, PushMode pushMode );

private:

    void setupProcessor( const std::vector<std::string> & schema,
                         const std::set<std::string> & neededColumns,
                         std::optional<std::string> symbolColumn,
                         bool subscribeAllOnEmptySymbol );

    bool readNextRow();

    // Option A: each subscriber owns its own field_map / target type,
    // so per-subscriber row conversion (Stage 4) can dispatch without
    // needing the manager to know Python.
    //   - monostate  -> return whole row as dict
    //   - string     -> single-column adapter (value from that column)
    //   - DictionaryPtr -> struct field map (csv column name -> struct field)
    //
    // m_dispatch is bound after the schema is known (bindSubscriberDispatchers).
    // The manager just calls it per row — the callback owns the conversion.
    // A null m_dispatch means "conversion not supported yet in this stage" and
    // the manager silently skips it.
    struct Subscriber {
        ManagedSimInputAdapter *                                          m_adapter;
        std::variant<std::monostate, std::string, DictionaryPtr>          m_fieldMap;
        std::function<void(const std::vector<std::string_view> &)>        m_dispatch;
    };

    // Walk registered subscribers and bind each one's m_dispatch based on its
    // field_map + the parsed header. Must be called after m_columnNames is populated.
    void bindSubscriberDispatchers();

    using dateTimeParserfn = DateTime(*)(std::string_view);

    // Registration-phase state (populated by getInputAdapter before start)
    std::vector<Subscriber> m_subscribers;
    std::unordered_map<std::string, std::vector<Subscriber>> m_subscribersBySymbol;

    // Configuration (from properties dict)
    csp::DateTime                       m_startTime;
    csp::DateTime                       m_endTime;
    std::string                         m_timeColumn;
    std::string                         m_timeFormat;
    std::string                         m_filename;
    std::string                         m_delimiter;
    std::string                         m_symbolColumnName; // configured symbol column name ("" = no symbol column)
    bool                                m_hasHeader;
    dateTimeParserfn                    dateParser;

    // Runtime state (initialized in start, used in processNextSimTimeSlice)
    std::vector<std::string>                    m_columnNames; // full header, in order
    std::optional<int>                          m_symbolColumn; // Index of symbol column
    std::vector<int>                            m_schema; // Indices to be used
    int                                         m_timeColumnIndex;
    std::ifstream                               m_file;
    std::string                                 m_row; // Current cached row
};

}

#endif // _IN_CSP_ADAPTERS_CSV_CsvInputAdapterManager_H
