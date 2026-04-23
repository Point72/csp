// RecordBatchRowProcessor implementation.

#include <csp/adapters/arrow/RecordBatchRowProcessor.h>
#include <csp/core/Exception.h>

namespace csp::adapters::arrow
{

std::set<std::string> RecordBatchRowProcessor::setupFromSchema(
    const std::shared_ptr<::arrow::Schema> & schema,
    const std::set<std::string> & columns,
    bool allowMissing,
    const std::unordered_map<std::string, std::shared_ptr<const StructMeta>> & structMetaByColumn )
{
    m_dispatchers.clear();
    m_dispatchers.reserve( columns.size() );
    m_nameToDispatcher.clear();
    m_schema = schema;
    m_numRows = 0;
    m_currentRow = 0;

    std::set<std::string> created;

    for( auto & col : columns )
    {
        int idx = schema -> GetFieldIndex( col );
        if( idx < 0 )
        {
            CSP_TRUE_OR_THROW_RUNTIME( allowMissing,
                "Missing column " << col << " in record batch schema" );
            continue;
        }

        auto arrowField = schema -> field( idx );

        // Look up StructMeta for STRUCT columns
        std::shared_ptr<const StructMeta> meta;
        auto metaIt = structMetaByColumn.find( col );
        if( metaIt != structMetaByColumn.end() )
            meta = metaIt -> second;

        auto dispatcher = createColumnDispatcher( arrowField, meta );
        if( !dispatcher )
            continue;  // unsupported type (e.g. STRUCT without meta) — skip silently

        m_nameToDispatcher[col] = dispatcher.get();
        m_dispatchers.push_back( std::move( dispatcher ) );
        created.insert( col );
    }

    return created;
}

void RecordBatchRowProcessor::bindBatch( const ::arrow::RecordBatch & batch )
{
    m_numRows = batch.num_rows();
    m_currentRow = 0;

    for( auto & dispatcher : m_dispatchers )
    {
        int idx = batch.schema() -> GetFieldIndex( dispatcher -> columnName() );
        CSP_TRUE_OR_THROW_RUNTIME( idx >= 0,
            "Column '" << dispatcher -> columnName() << "' not found in batch schema" );
        dispatcher -> bindColumn( batch.column( idx ).get() );
    }
}

bool RecordBatchRowProcessor::readNextRow()
{
    if( m_currentRow >= m_numRows ) [[unlikely]]
        return false;

    for( auto & dispatcher : m_dispatchers )
        dispatcher -> readNextValue();

    ++m_currentRow;
    return true;
}

void RecordBatchRowProcessor::dispatchRow( const utils::Symbol * symbol )
{
    for( auto & dispatcher : m_dispatchers )
        dispatcher -> dispatchValue( symbol );
}

bool RecordBatchRowProcessor::skipRow()
{
    if( m_currentRow >= m_numRows ) [[unlikely]]
        return false;

    for( auto & dispatcher : m_dispatchers )
        dispatcher -> fieldReader().skipNext();

    ++m_currentRow;
    return true;
}

void RecordBatchRowProcessor::addSubscriber(
    const std::string & column,
    ManagedSimInputAdapter * adapter,
    std::optional<utils::Symbol> symbol )
{
    auto it = m_nameToDispatcher.find( column );
    CSP_TRUE_OR_THROW_RUNTIME( it != m_nameToDispatcher.end(),
        "Cannot add subscriber: column '" << column << "' has no dispatcher" );
    it -> second -> addSubscriber( adapter, symbol );
}

bool RecordBatchRowProcessor::hasColumn( const std::string & name ) const
{
    return m_nameToDispatcher.find( name ) != m_nameToDispatcher.end();
}

ColumnDispatcher * RecordBatchRowProcessor::getDispatcher( const std::string & name )
{
    auto it = m_nameToDispatcher.find( name );
    return it != m_nameToDispatcher.end() ? it -> second : nullptr;
}

} // namespace csp::adapters::arrow
