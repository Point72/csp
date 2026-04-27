// Row-by-row RecordBatch processor using ColumnDispatchers.
// Owns a set of ColumnDispatchers (one per data column) and provides
// row reading and value dispatch via bound RecordBatchReaders.
//
// Usage: setupFromSchema() → bindSources() → readRowAndAdvance()/skipRow() → dispatchRow()
//
// Schema change handling is external: the caller detects schema changes
// and calls setupFromSchema() + re-registers subscribers.

#ifndef _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H
#define _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H

#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <arrow/record_batch.h>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace csp::adapters::arrow
{

class RecordBatchRowProcessor
{
public:
    struct ColumnMapping
    {
        std::string name;
        int         colIndex;
    };

    struct SourceEntry
    {
        ::arrow::RecordBatchReader *                  source = nullptr;
        std::shared_ptr<::arrow::RecordBatch>        currentBatch;
        int64_t                                      numRows    = 0;
        int64_t                                      currentRow = 0;
        std::vector<int>                             colIndices;
        std::vector<ColumnDispatcher *>              dispatchers;
    };

    RecordBatchRowProcessor() = default;

    // (Re)create dispatchers for the given schema and requested columns.
    std::set<std::string> setupFromSchema(
        const std::shared_ptr<::arrow::Schema> & schema,
        const std::set<std::string> & columns,
        bool allowMissing = false,
        const std::unordered_map<std::string, std::shared_ptr<const StructMeta>> & structMetaByColumn = {} );

    // Register a subscriber on a specific column.
    void addSubscriber( const std::string & column,
                        ManagedSimInputAdapter * adapter,
                        std::optional<utils::Symbol> symbol );

    // Check if a column has a dispatcher.
    bool hasColumn( const std::string & name ) const;
    ColumnDispatcher * getDispatcher( const std::string & name );

    // Dispatch current values to all subscribers.
    void dispatchRow( const utils::Symbol * symbol );

    // Bind sources. mappings[i] = list of {columnName, colIndex} for sources[i].
    void bindSources(
        const std::vector<::arrow::RecordBatchReader *> & sources,
        const std::vector<std::vector<ColumnMapping>> & mappings );

    // Skip one row across all sources. Returns false if EOF.
    bool skipRow();

    // Read all columns at current row and advance. Returns false if EOF.
    bool readRowAndAdvance();

private:
    std::vector<std::unique_ptr<ColumnDispatcher>>      m_dispatchers;
    std::unordered_map<std::string, ColumnDispatcher *>  m_nameToDispatcher;
    std::vector<SourceEntry>                             m_sources;

    bool ensureBatch( SourceEntry & entry );
    void rebindSource( SourceEntry & entry );
};

} // namespace csp::adapters::arrow

#endif // _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H
