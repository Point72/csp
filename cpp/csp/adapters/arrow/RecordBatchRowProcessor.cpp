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

        std::shared_ptr<const StructMeta> meta;
        auto metaIt = structMetaByColumn.find( col );
        if( metaIt != structMetaByColumn.end() )
            meta = metaIt -> second;

        auto dispatcher = createColumnDispatcher( arrowField, meta );
        if( !dispatcher )
            continue;

        m_nameToDispatcher[col] = dispatcher.get();
        m_dispatchers.push_back( std::move( dispatcher ) );
        created.insert( col );
    }

    return created;
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

void RecordBatchRowProcessor::dispatchRow( const utils::Symbol * symbol )
{
    for( auto & dispatcher : m_dispatchers )
        dispatcher -> dispatchValue( symbol );
}

void RecordBatchRowProcessor::bindSources(
    const std::vector<::arrow::RecordBatchReader *> & sources,
    const std::vector<std::vector<ColumnMapping>> & mappings )
{
    m_sources.clear();

    CSP_TRUE_OR_THROW_RUNTIME( sources.size() == mappings.size(),
        "bindSources: sources and mappings vectors must have the same size" );

    for( size_t srcIdx = 0; srcIdx < sources.size(); ++srcIdx )
    {
        SourceEntry entry;
        entry.source = sources[srcIdx];

        auto & colMappings = mappings[srcIdx];
        for( size_t j = 0; j < colMappings.size(); ++j )
        {
            auto & cm = colMappings[j];
            auto it = m_nameToDispatcher.find( cm.name );
            if( it == m_nameToDispatcher.end() )
                continue;

            entry.colIndices.push_back( cm.colIndex );
            entry.dispatchers.push_back( it -> second );
        }

        // Pull first non-empty batch
        for( ;; )
        {
            auto status = entry.source -> ReadNext( &entry.currentBatch );
            CSP_TRUE_OR_THROW_RUNTIME( status.ok(),
                "bindSources: failed to read first batch: " << status.ToString() );

            if( !entry.currentBatch )
                break;

            entry.numRows    = entry.currentBatch -> num_rows();
            entry.currentRow = 0;
            rebindSource( entry );

            if( entry.numRows > 0 )
                break;
        }

        m_sources.push_back( std::move( entry ) );
    }
}

bool RecordBatchRowProcessor::fetchNextBatch( SourceEntry & entry )
{
    while( entry.currentRow >= entry.numRows )
    {
        auto status = entry.source -> ReadNext( &entry.currentBatch );
        CSP_TRUE_OR_THROW_RUNTIME( status.ok(),
            "ensureBatch: failed to read next batch: " << status.ToString() );

        if( !entry.currentBatch )
            return false;

        entry.numRows    = entry.currentBatch -> num_rows();
        entry.currentRow = 0;
        rebindSource( entry );
    }
    return true;
}

void RecordBatchRowProcessor::rebindSource( SourceEntry & entry )
{
    for( size_t i = 0; i < entry.dispatchers.size(); ++i )
        entry.dispatchers[i] -> bindColumn( entry.currentBatch -> column( entry.colIndices[i] ).get() );
}

bool RecordBatchRowProcessor::skipRow()
{
    size_t successes = 0;
    for( auto & entry : m_sources )
    {
        if( ensureBatch( entry ) )
        {
            entry.currentRow++;
            ++successes;
        }
    }
    CSP_TRUE_OR_THROW_RUNTIME( successes == 0 || successes == m_sources.size(),
        "Input sources are not aligned - some have more data than others" );
    return successes > 0;
}

bool RecordBatchRowProcessor::readRowAndAdvance()
{
    size_t successes = 0;
    for( auto & entry : m_sources )
    {
        if( ensureBatch( entry ) )
        {
            for( size_t i = 0; i < entry.dispatchers.size(); ++i )
                entry.dispatchers[i] -> readValueAt( entry.currentRow );

            entry.currentRow++;
            ++successes;
        }
    }
    CSP_TRUE_OR_THROW_RUNTIME( successes == 0 || successes == m_sources.size(),
        "Input sources are not aligned - some have more data than others" );
    return successes > 0;
}

} // namespace csp::adapters::arrow
