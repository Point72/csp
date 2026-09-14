#ifndef _IN_CSP_ADAPTERS_CSV_CsvInputAdapterManager_H
#define _IN_CSP_ADAPTERS_CSV_CsvInputAdapterManager_H

#include "csp/core/Time.h"
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <fstream>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace csp::adapters::csv {

class CsvInputAdapterManager final : public csp::AdapterManager {
public:
  CsvInputAdapterManager(csp::Engine *engine, const Dictionary &properties);

  ~CsvInputAdapterManager();

  const char *name() const override { return "CsvInputAdapterManager"; }

  void start(DateTime starttime, DateTime endtime) override;
  void stop() override;
  DateTime processNextSimTimeSlice(DateTime time) override;

  ManagedSimInputAdapter *getInputAdapter(CspTypePtr &type,
                                          const Dictionary &properties,
                                          PushMode pushMode);

private:
  void setupProcessor(const std::vector<std::string> &schema,
                      const std::set<std::string> &neededColumns,
                      std::optional<std::string> symbolColumn,
                      bool subscribeAllOnEmptySymbol);

  bool readNextRow();

  struct StructSubscription {
    using FieldSetter =
        std::function<void(StructPtr &, const std::vector<std::string_view> &)>;

    ManagedSimInputAdapter *m_adapter;
    std::shared_ptr<const StructMeta> m_structMeta;

    std::vector<FieldSetter> m_fieldSetters;

    void dispatchValue(const std::vector<std::string_view> &cols) {
      StructPtr value = m_structMeta->create();

      for (auto &setter : m_fieldSetters)
        setter(value, cols);

      m_adapter->pushTick(value);
    }
  };

  struct Subscriber {
    ManagedSimInputAdapter *m_adapter;
    std::variant<std::monostate, std::string, DictionaryPtr> m_fieldMap;

    std::function<void(const std::vector<std::string_view> &)> m_dispatch;
    std::optional<StructSubscription> m_structSubscription;
  };

  // Walk registered subscribers and bind each one's m_dispatch based on its
  // field_map + the parsed header. Must be called after m_columnNames is
  // populated.
  void bindSubscriberDispatchers();

  using dateTimeParserfn = std::function<DateTime(std::string_view)>;

  // Registration-phase state (populated by getInputAdapter before start)
  std::vector<Subscriber> m_subscribers;
  std::unordered_map<std::string, std::vector<Subscriber>>
      m_subscribersBySymbol;

  // Configuration (from properties dict)
  csp::DateTime m_startTime;
  csp::DateTime m_endTime;
  std::string m_timeColumn;
  std::string m_timeFormat;
  std::string m_filename;
  std::string m_delimiter;
  std::string m_symbolColumnName; // configured symbol column name ("" = no
                                  // symbol column)
  bool m_hasHeader;
  dateTimeParserfn dateParser;

  // Runtime state (initialized in start, used in processNextSimTimeSlice)
  std::vector<std::string> m_columnNames; // full header, in order
  std::optional<int> m_symbolColumn;      // Index of symbol column
  std::vector<int> m_schema;              // Indices to be used
  int m_timeColumnIndex;
  std::ifstream m_file;
  std::string m_row; // Current cached row
};

} // namespace csp::adapters::csv

#endif // _IN_CSP_ADAPTERS_CSV_CsvInputAdapterManager_H
