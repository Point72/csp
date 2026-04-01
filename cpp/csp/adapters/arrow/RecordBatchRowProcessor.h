// Row-by-row RecordBatch processor using ColumnDispatchers.
// Owns a set of ColumnDispatchers (one per data column) and provides
// uniform batch binding, row reading, and value dispatch.  Designed to
// be shared by both the parquet adapter manager (with time/symbol on top)
// and a future arrow adapter manager.
//
// Schema change handling is external: the caller detects schema changes
// and calls setupFromSchema() + re-registers subscribers.

#ifndef _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H
#define _IN_CSP_ADAPTERS_ARROW_RecordBatchRowProcessor_H

#include <csp/adapters/arrow/ColumnDispatcher.h>
#include <arrow/record_batch.h>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace csp::adapters::arrow
{

class RecordBatchRowProcessor
{
public:
    RecordBatchRowProcessor() = default;

    // (Re)create dispatchers for the given schema and requested columns.
    // Clears any existing dispatchers and subscribers.
    // Columns not found in the schema are skipped if allowMissing is true,
    // otherwise an exception is thrown.
    // structMetaByColumn provides StructMeta for STRUCT-typed columns so they
    // can be created in a single pass (without deferred addDispatcher).
    // Returns the set of column names for which dispatchers were created.
    std::set<std::string> setupFromSchema(
        const std::shared_ptr<::arrow::Schema> & schema,
        const std::set<std::string> & columns,
        bool allowMissing = false,
        const std::unordered_map<std::string, std::shared_ptr<const StructMeta>> & structMetaByColumn = {} );

    // Bind all dispatchers to columns from a new RecordBatch.
    void bindBatch( const ::arrow::RecordBatch & batch );

    // Read the next row's values into all dispatchers.
    // Returns false if no more rows in the current batch.
    bool readNextRow();

    // Dispatch current values for all dispatchers.
    void dispatchRow( const utils::Symbol * symbol );

    // Advance all dispatchers without reading/dispatching (skip row).
    // Returns false if no more rows in the current batch.
    bool skipRow();

    // Register a subscriber on a specific column.
    void addSubscriber( const std::string & column,
                        ManagedSimInputAdapter * adapter,
                        std::optional<utils::Symbol> symbol );

    // Check if a column has a dispatcher.
    bool hasColumn( const std::string & name ) const;

    // Get dispatcher for a column (nullptr if not found).
    ColumnDispatcher * getDispatcher( const std::string & name );

    int64_t numRows() const { return m_numRows; }
    int64_t currentRow() const { return m_currentRow; }
    bool hasMoreRows() const { return m_currentRow < m_numRows; }

    // Schema from last setupFromSchema call.
    const std::shared_ptr<::arrow::Schema> & schema() const { return m_schema; }

private:
    std::vector<std::unique_ptr<ColumnDispatcher>>   m_dispatchers;
    std::unordered_map<std::string, ColumnDispatcher *> m_nameToDispatcher;
    std::shared_ptr<::arrow::Schema>                 m_schema;
    int64_t                                          m_numRows    = 0;
    int64_t                                          m_currentRow = 0;
};

} // namespace csp::adapters::arrow

#endif
