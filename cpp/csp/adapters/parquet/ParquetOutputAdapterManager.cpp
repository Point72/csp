#include <csp/adapters/parquet/ParquetOutputAdapterManager.h>
#include <csp/adapters/parquet/ParquetOutputFilenameAdapter.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ParquetDictBasketOutputWriter.h>
#include <csp/adapters/parquet/ParquetWriter.h>
#include <csp/engine/Dictionary.h>

namespace csp::adapters::parquet
{


ParquetOutputAdapterManager::ParquetOutputAdapterManager( csp::Engine *engine, const Dictionary &properties, FileVisitorCallback fileVisitor ) :
    AdapterManager( engine ), m_fileVisitor( fileVisitor ), m_outputFilenameAdapter( nullptr )
{
    m_fileName            = properties.get<std::string>( "file_name" );
    m_timestampColumnName = properties.get<std::string>( "timestamp_column_name" );
    m_allowOverwrite      = properties.get<bool>( "allow_overwrite" );
    m_batchSize           = properties.get<std::int64_t>( "batch_size" );
    m_compression         = properties.get<std::string>( "compression" );
    m_writeArrowBinary    = properties.get<bool>( "write_arrow_binary" );
    m_splitColumnsToFiles = properties.get<bool>( "split_columns_to_files" );
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
    bool visitFile = m_fileVisitor && m_parquetWriter -> isFileOpen();
    m_parquetWriter -> stop();
    m_parquetWriter = nullptr;

    for( auto &&writer:m_dictBasketWriters )
    {
        writer -> stop();
    }
    m_dictBasketWriters.clear();

    if( visitFile )
        m_fileVisitor( m_fileName );
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

OutputAdapter *ParquetOutputAdapterManager::getListOutputAdapter( CspTypePtr &elemType, const Dictionary &properties,
                                                                  const DialectGenericListWriterInterface::Ptr& listWriterInterface)
{
    auto columnName = properties.get<std::string>( "column_name" );
    return m_parquetWriter -> getListOutputAdapter( elemType, columnName, listWriterInterface );
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

    m_dictBasketWriterIndexByName[ columnName ] = m_dictBasketWriters.size() - 1;
    return m_dictBasketWriters.back().get();
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

    if( m_fileVisitor )
        m_fileVisitor( m_fileName );

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
