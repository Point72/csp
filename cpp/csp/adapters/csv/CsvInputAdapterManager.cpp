#include "csp/core/Exception.h"
#include <array>
#include <charconv>
#include <csp/adapters/csv/CsvInputAdapterManager.h>
#include <csp/adapters/utils/ValueDispatcher.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/CspType.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <ranges>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

namespace csp::adapters::csv {

// Build parser at runtime for extracting date time
std::function<DateTime(std::string_view)>
createParser(std::string_view format) {

  constexpr std::array<std::string_view, 6> date_format = {"YYYY", "MM", "DD",
                                                           "hh",   "mm", "ss"};

  std::array<int, 6> date_indices;

  for (int i = 0; i < 6; ++i) {
    auto pos = format.find(date_format[i]);

    date_indices[i] =
        pos == std::string_view::npos ? -1 : static_cast<int>(pos);
  }

  return [date_indices](std::string_view date) -> DateTime {
    std::array<int, 6> data = {
        0, // year
        0, // month
        0, // day
        0, // hour
        0, // minute
        0  // second
    };

    auto requireLength = [&](std::string_view date, int idx, int width) {
      CSP_TRUE_OR_THROW_RUNTIME(idx >= 0 && static_cast<size_t>(idx + width) <=
                                                date.size(),
                                "Invalid timestamp '" << date << "'");
    };

    auto d2 = [&](int idx) {
      return (date[idx] - '0') * 10 + (date[idx + 1] - '0');
    };

    auto d4 = [&](int idx) {
      return (date[idx] - '0') * 1000 + (date[idx + 1] - '0') * 100 +
             (date[idx + 2] - '0') * 10 + (date[idx + 3] - '0');
    };

    for (int i = 0; i < 6; ++i) {
      if (date_indices[i] == -1)
        continue;

      if (i == 0) {
        requireLength(date, date_indices[i], 4);
        data[i] = d4(date_indices[i]);
      } else {
        requireLength(date, date_indices[i], 2);
        data[i] = d2(date_indices[i]);
      }
    }

    return DateTime(data[0], data[1], data[2], data[3], data[4], data[5]);
  };
}

CsvInputAdapterManager::CsvInputAdapterManager(csp::Engine *engine,
                                               const Dictionary &properties)
    : AdapterManager(engine) {
  m_filename = properties.get<std::string>("filename", "");
  CSP_TRUE_OR_THROW_RUNTIME(!m_filename.empty(), "Filename must be provided");

  auto tz = properties.get<std::string>("tz", "UTC");
  CSP_TRUE_OR_THROW_RUNTIME(
      tz == "UTC", "Only UTC default timezone is supported, got:" << tz);

  properties.tryGet("start_time", m_startTime);
  properties.tryGet("end_time", m_endTime);

  m_delimiter = properties.get<std::string>("delimiter", ",");
  m_hasHeader = properties.get<bool>("hasHeader", true);
  m_timeColumn = properties.get<std::string>("time_column", "");
  m_symbolColumnName = properties.get<std::string>("symbol_column", "");

  CSP_TRUE_OR_THROW_RUNTIME(m_timeColumn != "", "Time column can't be empty");
  CSP_TRUE_OR_THROW_RUNTIME(!m_delimiter.empty(),
                            "CSV delimiter cannot be empty");

  properties.tryGet("time_format", m_timeFormat);

  if (m_timeFormat.empty()) {
    CSP_THROW(ValueError, "Time format must be provided");
  }

  dateParser = createParser(m_timeFormat);
}

CsvInputAdapterManager::~CsvInputAdapterManager() = default;

void CsvInputAdapterManager::setupProcessor(
    const std::vector<std::string> &schema,
    const std::set<std::string> &neededColumns,
    std::optional<std::string> symbolColumn, bool subscribeAllOnEmptySymbol) {
  m_schema = {};
  m_symbolColumn = std::nullopt;

  for (int i = 0; i < std::ssize(schema); ++i) {
    if (neededColumns.contains(schema[i])) {
      m_schema.push_back(i);
    }

    if (symbolColumn && *symbolColumn == schema[i]) [[unlikely]] {
      m_symbolColumn = i;
    }
  }
}

ManagedSimInputAdapter *CsvInputAdapterManager::getInputAdapter(
    CspTypePtr &type, const Dictionary &properties, PushMode pushMode) {
  // Per-subscription symbol filter. Empty string means "subscribe to every
  // row".
  std::string symbol = properties.get<std::string>("symbol", "");

  auto *adapter =
      engine()->createOwnedObject<ManagedSimInputAdapter>(type, this, pushMode);

  Subscriber sub;
  sub.m_adapter = adapter;

  // Stash the field_map so the subscriber can convert its row at dispatch time.
  // string -> single-column; DictionaryPtr -> struct field map;
  // absent/None -> whole-row dict.
  if (properties.exists("field_map")) {
    auto &fm = properties.getUntypedValue("field_map");
    if (std::holds_alternative<std::string>(fm))
      sub.m_fieldMap = std::get<std::string>(fm);
    else if (std::holds_alternative<DictionaryPtr>(fm))
      sub.m_fieldMap = std::get<DictionaryPtr>(fm);
    // else: leave monostate — whole-row dict output
  }

  if (symbol.empty())
    m_subscribers.push_back(std::move(sub));
  else
    m_subscribersBySymbol[symbol].push_back(std::move(sub));

  return adapter;
}

void CsvInputAdapterManager::start(DateTime starttime, DateTime endtime) {
  if (!m_startTime.isNone()) {
    starttime = std::max(starttime, m_startTime);
  }
  AdapterManager::start(starttime, endtime);

  m_file = std::ifstream(m_filename, std::ios::binary);
  if (!m_file)
    CSP_THROW(IOError, "Failed to open " << m_filename);

  // Reusable split helper: split a line by m_delimiter into owned strings,
  // trimming a trailing '\r' from the final field for Windows CSVs.
  auto splitLine = [](std::string_view line, std::string_view delim) {
    std::vector<std::string> parts;
    for (auto part : std::views::split(line, delim)) {
      auto begin = part.begin();
      auto len = std::ranges::distance(part);
      parts.emplace_back(len == 0 ? std::string() : std::string(&*begin, len));
    }
    if (!parts.empty() && !parts.back().empty() && parts.back().back() == '\r')
      parts.back().pop_back();
    return parts;
  };

  // Parse header
  m_columnNames.clear();
  if (m_hasHeader) {
    std::string headerLine;
    if (!std::getline(m_file, headerLine))
      CSP_THROW(IOError, "Failed to read header from " << m_filename);
    m_columnNames = splitLine(headerLine, m_delimiter);
  }

  // Ensure no duplicate column names
  std::unordered_set<std::string> seen;

  for (const auto &name : m_columnNames) {
    CSP_TRUE_OR_THROW_RUNTIME(seen.insert(name).second,
                              "Duplicate CSV column name: '" << name << "'");
  }

  // Collect the set of columns any subscriber cares about
  std::set<std::string> neededColumns;
  bool needAllColumns = false;

  if (!m_timeColumn.empty())
    neededColumns.insert(m_timeColumn);
  if (!m_symbolColumnName.empty())
    neededColumns.insert(m_symbolColumnName);

  auto collectFrom = [&](const Subscriber &sub) {
    if (std::holds_alternative<std::monostate>(sub.m_fieldMap)) {
      needAllColumns = true;
    } else if (std::holds_alternative<std::string>(sub.m_fieldMap)) {
      neededColumns.insert(std::get<std::string>(sub.m_fieldMap));
    } else if (std::holds_alternative<DictionaryPtr>(sub.m_fieldMap)) {
      auto &fm = std::get<DictionaryPtr>(sub.m_fieldMap);
      for (auto it = fm->begin(); it != fm->end(); ++it)
        neededColumns.insert(it.key());
    }
  };

  for (const auto &sub : m_subscribers)
    collectFrom(sub);
  for (const auto &[symbol, subs] : m_subscribersBySymbol)
    for (const auto &sub : subs)
      collectFrom(sub);

  if (needAllColumns)
    for (const auto &name : m_columnNames)
      neededColumns.insert(name);

  // Do column-name -> index mappings through setupProcessor
  std::optional<std::string> symbolColumnOpt;
  if (!m_symbolColumnName.empty())
    symbolColumnOpt = m_symbolColumnName;

  bool subscribeAllOnEmptySymbol = !m_subscribers.empty();
  setupProcessor(m_columnNames, neededColumns, symbolColumnOpt,
                 subscribeAllOnEmptySymbol);

  // Locate the time column so processNextSimTimeSlice can extract it O(1).
  m_timeColumnIndex = -1;
  for (int i = 0; i < std::ssize(m_columnNames); ++i) {
    if (m_columnNames[i] == m_timeColumn) {
      m_timeColumnIndex = i;
      break;
    }
  }

  CSP_TRUE_OR_THROW_RUNTIME(m_timeColumnIndex >= 0,
                            "Time column '" << m_timeColumn
                                            << "' not found in CSV header");

  if (!m_symbolColumnName.empty()) {
    CSP_TRUE_OR_THROW_RUNTIME(m_symbolColumn.has_value(),
                              "Symbol column '" << m_symbolColumnName
                                                << "' not found in CSV header");
  }

  // Bind each subscriber's row -> tick callback now that the schema is known.
  bindSubscriberDispatchers();

  // Cache the first data row so processNextSimTimeSlice's skip loop has data to
  // compare.
  if (!std::getline(m_file, m_row))
    m_row.clear();
}

static void setFieldFromCsv(StructPtr &s, const StructField &field,
                            std::string_view value) {
  switch (field.type()->type()) {
  case CspType::Type::STRING:
    field.setValue<std::string>(s.get(), std::string(value));
    break;

  case CspType::Type::INT64: {
    int64_t x;
    auto [p, ec] =
        std::from_chars(value.data(), value.data() + value.size(), x);

    CSP_TRUE_OR_THROW_RUNTIME(ec == std::errc{} &&
                                  p == value.data() + value.size(),
                              "Invalid int64: '" << value << "'");

    field.setValue<int64_t>(s.get(), x);
    break;
  }

  case CspType::Type::DOUBLE:
    field.setValue<double>(s.get(), std::stod(std::string(value)));
    break;

  default:
    CSP_THROW(TypeError, "Unsupported CSV type");
  }
}

void CsvInputAdapterManager::bindSubscriberDispatchers() {
  // Column name -> header index (built once).
  std::unordered_map<std::string_view, size_t> colIndex;
  colIndex.reserve(m_columnNames.size());
  for (size_t i = 0; i < m_columnNames.size(); ++i)
    colIndex[m_columnNames[i]] = i;

  auto bind = [&](Subscriber &sub) {
    // Whole-row dict: Build a struct
    if (std::holds_alternative<std::monostate>(sub.m_fieldMap)) {
      auto *structType =
          static_cast<const CspStructType *>(sub.m_adapter->dataType());

      auto meta = structType->meta();

      StructSubscription subscription;
      subscription.m_adapter = sub.m_adapter;
      subscription.m_structMeta = meta;

      for (size_t i = 0; i < m_columnNames.size(); i++) {
        auto field = meta->field(m_columnNames[i]);

        if (!field)
          continue;

        subscription.m_fieldSetters.push_back(
            [i, field](StructPtr &s,
                       const std::vector<std::string_view> &cols) {
              setFieldFromCsv(s, *field, cols[i]);
            });
      }

      sub.m_structSubscription = std::move(subscription);
      return;
    }

    // Single-column subscription — extract the named column and push it.
    if (std::holds_alternative<std::string>(sub.m_fieldMap)) {
      const auto &colName = std::get<std::string>(sub.m_fieldMap);
      auto it = colIndex.find(colName);
      CSP_TRUE_OR_THROW_RUNTIME(it != colIndex.end(),
                                "Column '" << colName
                                           << "' not found in CSV header");
      size_t idx = it->second;

      auto *adapter = sub.m_adapter;
      auto tag = adapter->dataType()->type();

      if (tag == CspType::Type::STRING) {
        sub.m_dispatch = [adapter,
                          idx](const std::vector<std::string_view> &cols) {
          adapter->pushTick<std::string>(std::string(cols[idx]));
        };
      } else if (tag == CspType::Type::INT64) {
        sub.m_dispatch = [adapter,
                          idx](const std::vector<std::string_view> &cols) {
          int64_t value = 0;
          auto [ptr, ec] = std::from_chars(
              cols[idx].data(), cols[idx].data() + cols[idx].size(), value);

          CSP_TRUE_OR_THROW_RUNTIME(
              ec == std::errc{} && ptr == cols[idx].data() + cols[idx].size(),
              "Failed to parse int64 CSV value '" << cols[idx] << "'");

          adapter->pushTick<int64_t>(value);
        };
      } else if (tag == CspType::Type::DOUBLE) {
        sub.m_dispatch = [adapter,
                          idx](const std::vector<std::string_view> &cols) {
          // std::from_chars not implemented for double types on Clang
          double value = std::stod(std::string(cols[idx]));
          adapter->pushTick<double>(value);
        };
      }
      // else: leave m_dispatch null;
      return;
    }

    if (std::holds_alternative<DictionaryPtr>(sub.m_fieldMap)) {
      auto *structType =
          static_cast<const CspStructType *>(sub.m_adapter->dataType());

      auto meta = structType->meta();

      StructSubscription subscription;
      subscription.m_adapter = sub.m_adapter;
      subscription.m_structMeta = meta;

      auto &fm = std::get<DictionaryPtr>(sub.m_fieldMap);

      for (auto it = fm->begin(); it != fm->end(); ++it) {
        const std::string csvColumn = it.key();
        const std::string structField = it.value<std::string>();

        auto col = colIndex.find(csvColumn);

        CSP_TRUE_OR_THROW_RUNTIME(col != colIndex.end(),
                                  "Column '" << csvColumn
                                             << "' not found in CSV header");

        auto field = meta->field(structField);

        CSP_TRUE_OR_THROW_RUNTIME(field, "Field '" << structField
                                                   << "' not found in struct");

        size_t idx = col->second;

        subscription.m_fieldSetters.push_back(
            [idx, field](StructPtr &s,
                         const std::vector<std::string_view> &cols) {
              setFieldFromCsv(s, *field, cols[idx]);
            });
      }

      sub.m_structSubscription = std::move(subscription);
      return;
    }
  };

  for (auto &sub : m_subscribers)
    bind(sub);
  for (auto &[symbol, subs] : m_subscribersBySymbol)
    for (auto &sub : subs)
      bind(sub);
}

void CsvInputAdapterManager::stop() {
  m_subscribers.clear();
  m_subscribersBySymbol.clear();
  m_schema.clear();
  m_file.close();
  AdapterManager::stop();
}

DateTime CsvInputAdapterManager::processNextSimTimeSlice(DateTime time) {
  if (m_row.empty()) [[unlikely]]
    return DateTime::NONE();

  // Split m_row once per row into string_views. Views are valid until the
  // next getline() mutates m_row, so every read of `cols` must precede the
  // next read from the file.
  auto splitRow = [this]() {
    std::vector<std::string_view> cols;
    for (auto part : std::views::split(m_row, m_delimiter)) {
      auto begin = part.begin();
      auto len = std::ranges::distance(part);
      cols.emplace_back(len == 0 ? std::string_view()
                                 : std::string_view(&*begin, len));
    }
    // Trim trailing '\r' on Windows CSVs so the final field parses cleanly.
    if (!cols.empty() && !cols.back().empty() && cols.back().back() == '\r')
      cols.back().remove_suffix(1);
    return cols;
  };

  // Skip loop: advance until we find a row at or after `time`.
  std::vector<std::string_view> cols = splitRow();
  DateTime rowTime = dateParser(cols[m_timeColumnIndex]);
  while (rowTime < time) {
    if (!std::getline(m_file, m_row)) {
      m_row.clear();
      return DateTime::NONE();
    }
    cols = splitRow();
    rowTime = dateParser(cols[m_timeColumnIndex]);
  }

  if (!m_endTime.isNone() && rowTime > m_endTime)
    return DateTime::NONE();

  if (rowTime > time)
    return rowTime;

  // Dispatch every row with this exact timestamp.
  do {
    // Subscribe-all subscribers see every row.
    for (auto &sub : m_subscribers) {
      if (sub.m_structSubscription)
        sub.m_structSubscription->dispatchValue(cols);

      else if (sub.m_dispatch)
        sub.m_dispatch(cols);
    }

    // Symbol-filtered subscribers only see rows where their symbol matches.
    if (m_symbolColumn.has_value()) {
      std::string sym(cols[*m_symbolColumn]);
      auto it = m_subscribersBySymbol.find(sym);
      if (it != m_subscribersBySymbol.end())
        for (auto &sub : it->second) {
          if (sub.m_structSubscription)
            sub.m_structSubscription->dispatchValue(cols);

          else if (sub.m_dispatch)
            sub.m_dispatch(cols);
        }
    }

    if (!std::getline(m_file, m_row)) {
      m_row.clear();
      return DateTime::NONE();
    }
    cols = splitRow();
    rowTime = dateParser(cols[m_timeColumnIndex]);
  } while (rowTime == time);

  return rowTime;
}
} // namespace csp::adapters::csv
