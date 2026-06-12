#include <csp/adapters/parquet/ParquetOutputAdapterManager.h>
#include <csp/adapters/parquet/ParquetOutputFilenameAdapter.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ParquetDictBasketOutputWriter.h>
#include <csp/adapters/parquet/ParquetWriter.h>
#include <csp/engine/Dictionary.h>

namespace csp::adapters::parquet
{


ParquetOutputAdapterManager::ParquetOutputAdapterManager( csp::Engine *engine, const Dictionary &properties ) :
    AdapterManager( engine ), m_outputFilenameAdapter( nullptr )
{
    m_fileName            = properties.get<std::string>( "file_name" );
    m_timestampColumnName = properties.get<std::string>( "timestamp_column_name" );
    m_batchSize           = properties.get<std::int64_t>( "batch_size" );
    m_parquetWriter       = std::make_unique<ParquetWriter>( this, properties );
}

ParquetOutputAdapterManager::~ParquetOutputAdapterManager()
{
}

void ParquetOutputAdapterManager::start( DateTime starttime, DateTime endtime )
{
    m_parquetWriter -> start();
    for( auto &&writer:m_dictBasketWriters )
    {
        writer -> start();
    }
}

void ParquetOutputAdapterManager::stop()
{
    // Stop all writers (flush + close their files) before destroying any of them, so a basket
    // writer can never observe a destroyed sibling during teardown.
    m_parquetWriter -> stop();
    for( auto &&writer:m_dictBasketWriters )
    {
        writer -> stop();
    }
    m_parquetWriter = nullptr;
    m_dictBasketWriters.clear();
}

DateTime ParquetOutputAdapterManager::processNextSimTimeSlice( DateTime time )
{
    return DateTime::NONE();
}

OutputAdapter *ParquetOutputAdapterManager::getOutputAdapter( CspTypePtr &type, const Dictionary &properties )
{
    if( type -> type() == CspType::Type::STRUCT )
    {
        return getStructOutputAdapter( type, properties );
    }
    else
    {
        return getScalarOutputAdapter( type, properties );
    }
}

OutputAdapter *ParquetOutputAdapterManager::getListOutputAdapter( CspTypePtr &elemType, const Dictionary &properties )
{
    auto columnName = properties.get<std::string>( "column_name" );
    return m_parquetWriter -> getListOutputAdapter( elemType, columnName );
}


ParquetDictBasketOutputWriter *
ParquetOutputAdapterManager::createDictOutputBasketWriter( const char *columnName, const CspTypePtr &cspTypePtr )
{
    auto &&existingAdapterIt = m_dictBasketWriterIndexByName.find( columnName );
    CSP_TRUE_OR_THROW_RUNTIME( existingAdapterIt == m_dictBasketWriterIndexByName.end(),
                               "Trying to create output basket writer for " << columnName << " more than once" );

    if( cspTypePtr -> type() == CspType::Type::STRUCT )
    {
        m_dictBasketWriters.push_back( std::make_unique<ParquetStructDictBasketOutputWriter>( this, columnName, cspTypePtr ) );
    }
    else
    {
        m_dictBasketWriters.push_back( std::make_unique<ParquetScalarDictBasketOutputWriter>( this, columnName, cspTypePtr ) );
    }

    // Provide data sink via factory if available
    auto * writer = m_dictBasketWriters.back().get();
    if( m_sinkFactory )
    {
        writer -> setSink( m_sinkFactory( columnName ) );
        std::string indexName = std::string( columnName ) + "__csp_index";
        writer -> setIndexSink( m_sinkFactory( indexName ) );
    }

    m_dictBasketWriterIndexByName[ columnName ] = m_dictBasketWriters.size() - 1;
    return writer;
}

OutputAdapter *ParquetOutputAdapterManager::createOutputFileNameAdapter()
{
    CSP_TRUE_OR_THROW_RUNTIME( m_outputFilenameAdapter == nullptr, "Trying to set output filename adapter more than once" );
    m_outputFilenameAdapter = engine() -> createOwnedObject<ParquetOutputFilenameAdapter>( *this );
    return m_outputFilenameAdapter;
}

void ParquetOutputAdapterManager::changeFileName( const std::string &filename )
{
    if( m_parquetWriter )
    {
        m_parquetWriter -> onFileNameChange( filename );
    }

    m_fileName = filename;
}

void ParquetOutputAdapterManager::scheduleEndCycle()
{
    if( rootEngine() -> scheduleEndCycleListener( m_parquetWriter.get() ) )
    {
        for( auto &&basketWriter:m_dictBasketWriters )
        {
            rootEngine() -> scheduleEndCycleListener( basketWriter.get() );
        }
    }
}

void ParquetOutputAdapterManager::setSink( RecordBatchSink sink )
{
    m_parquetWriter -> setSink( std::move( sink ) );
}

OutputAdapter *ParquetOutputAdapterManager::getScalarOutputAdapter( CspTypePtr &type, const Dictionary &properties )
{
    auto columnName = properties.get<std::string>( "column_name" );

    return m_parquetWriter -> getScalarOutputAdapter( type, columnName );
}

OutputAdapter *ParquetOutputAdapterManager::getStructOutputAdapter( CspTypePtr &type, const Dictionary &properties )
{
    auto fieldMap = properties.get<DictionaryPtr>( "field_map" );

    return m_parquetWriter -> getStructOutputAdapter( type, fieldMap );
}


}
