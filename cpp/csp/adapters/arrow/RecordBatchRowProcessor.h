// Row-by-row RecordBatch processor using ColumnDispatchers.

#ifndef _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H
#define _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H

#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <functional>
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
    // Function signature for reading the next batch from a source.
    using ReadNextFn = std::function<::arrow::Status( std::shared_ptr<::arrow::RecordBatch> * )>;

    struct ColumnMapping
    {
        std::string name;
        int         colIndex;
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

    // Bind batch sources.  Each source is a readNext function (wrapping an async generator
    // or sync reader).  mappings[i] = list of {columnName, colIndex} for sources[i].
    void bindSources(
        const std::vector<ReadNextFn> & sources,
        const std::vector<std::vector<ColumnMapping>> & mappings );

    // Skip one row across all sources. Returns false if EOF.
    bool skipRow();

    // Read all columns at current row and advance. Returns false if EOF.
    bool readRowAndAdvance();

private:
    struct SourceEntry
    {
        ReadNextFn                                       readNext;
        std::shared_ptr<::arrow::RecordBatch>            currentBatch;
        int64_t                                          numRows    = 0;
        int64_t                                          currentRow = 0;
        std::vector<int>                                 colIndices;
        std::vector<ColumnDispatcher *>                  dispatchers;
    };

    std::vector<std::unique_ptr<ColumnDispatcher>>      m_dispatchers;
    std::unordered_map<std::string, ColumnDispatcher *>  m_nameToDispatcher;
    std::vector<SourceEntry>                             m_sources;

    bool fetchNextBatch( SourceEntry & entry );
    void rebindSource( SourceEntry & entry );

    inline bool ensureBatch( SourceEntry & entry )
    {
        if( entry.currentRow < entry.numRows ) [[likely]]
            return true;
        return fetchNextBatch( entry );
    }
};

} // namespace csp::adapters::arrow

#endif // _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H
