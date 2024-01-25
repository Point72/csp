#include <csp/adapters/parquet/ParquetWriter.h>
#include <csp/adapters/parquet/ArrowIPCFileWriterWrapper.h>
#include <csp/adapters/parquet/FileWriterWrapperContainer.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/adapters/parquet/FileWriterWrapperContainer.h>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/io/file.h>
#include <arrow/util/key_value_metadata.h>
#include <parquet/arrow/writer.h>

namespace csp::adapters::parquet
{

ParquetWriter::ParquetWriter( ParquetOutputAdapterManager *mgr, std::optional<bool> writeTimestampColumn )
        : m_adapterMgr( *mgr ), m_engine( mgr -> engine() ), m_curChunkSize( 0 ), m_writeTimestampColumn( writeTimestampColumn )
{}

ParquetWriter::ParquetWriter( ParquetOutputAdapterManager *mgr, const Dictionary & properties ) : ParquetWriter( mgr, std::optional<bool>{} )
{
    if( properties.exists( "file_metadata" ) )
    {
        auto file_meta = properties.get<DictionaryPtr>( "file_metadata" );
        m_fileMetaData = std::make_shared<arrow::KeyValueMetadata>();
        for( auto it = file_meta -> begin(); it != file_meta -> end(); ++it )
        {
            const std::string * value = std::get_if<std::string>( &it.getUntypedValue() );
            if( !value )
                CSP_THROW( TypeError, "parquet metadata can only have string values" );
            m_fileMetaData -> Append( it.key(), *value );
        }
    }

    if( properties.exists( "column_metadata" ) )
    {
        auto column_metadata = properties.get<DictionaryPtr>( "column_metadata" );
        for( auto colIt = column_metadata -> begin(); colIt != column_metadata -> end(); ++colIt )
        {
            const DictionaryPtr * cmeta = std::get_if<DictionaryPtr>( &colIt.getUntypedValue() );
            if( !cmeta )
                CSP_THROW( TypeError, "parquet column metadata expects dictionary entry per column, got unrecognized type for column '" << colIt.key() << "'" );

            auto kv_metadata = std::make_shared<arrow::KeyValueMetadata>();

            for( auto it = (*cmeta) -> begin(); it != (*cmeta) -> end(); ++it )
            {
                const std::string * value = std::get_if<std::string>( &it.getUntypedValue() );
                if( !value )
                    CSP_THROW( TypeError, "parquet column metadata can only have string values, got non-string value for metadata on column '" << colIt.key() << "'" );
                kv_metadata -> Append( it.key(), *value );
            }

            m_columnMetaData[ colIt.key() ] = kv_metadata;
        }
    }
}

ParquetWriter::~ParquetWriter()
{
    if( m_curChunkSize > 0 )
    {
        writeCurChunkToFile();
    }
}

SingleColumnParquetOutputAdapter *ParquetWriter::getScalarOutputAdapter( CspTypePtr &type, const std::string &columnName )
{
    auto res = static_cast<SingleColumnParquetOutputAdapter *>(getScalarOutputHandler( type, columnName ));
    return res;
}

StructParquetOutputAdapter *ParquetWriter::getStructOutputAdapter( CspTypePtr &type, csp::DictionaryPtr fieldMap )
{
    auto res = static_cast<StructParquetOutputAdapter *>(getStructOutputHandler( type, fieldMap ));
    return res;
}

SingleColumnParquetOutputHandler *ParquetWriter::getScalarOutputHandler( CspTypePtr &type, const std::string &columnName )
{
    CSP_TRUE_OR_THROW_RUNTIME( m_publishedColumnNames.emplace( columnName ).second,
                               "Trying to publish column " << columnName << " more than once" );
    //create and register adapter
    auto adapter = createScalarOutputHandler( type, columnName );
    m_adapters.emplace_back( adapter );
    return adapter;
}

StructParquetOutputHandler *ParquetWriter::getStructOutputHandler( CspTypePtr &type, csp::DictionaryPtr fieldMap )
{
    for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
    {
        auto &&columnName = it.value<std::string>();
        CSP_TRUE_OR_THROW_RUNTIME( m_publishedColumnNames.emplace( columnName ).second,
                                   "Trying to publish column " << columnName << " more than once" );
    }

    auto adapter = createStructOutputHandler( type, fieldMap );
    m_adapters.emplace_back( adapter );

    return adapter;
}

ListColumnParquetOutputHandler *ParquetWriter::getListOutputHandler( CspTypePtr &elemType, const std::string &columnName,
                                                                     const DialectGenericListWriterInterface::Ptr &listWriterInterface )
{
    CSP_TRUE_OR_THROW_RUNTIME( m_publishedColumnNames.emplace( columnName ).second,
                               "Trying to publish column " << columnName << " more than once" );
    //create and register adapter
    auto adapter = createListOutputHandler( elemType, columnName,  listWriterInterface);
    m_adapters.emplace_back( adapter );
    return adapter;
}

ListColumnParquetOutputAdapter *ParquetWriter::getListOutputAdapter(
        CspTypePtr &elemType, const std::string &columnName,
        const DialectGenericListWriterInterface::Ptr &listWriterInterface )
{
    auto res = static_cast<ListColumnParquetOutputAdapter *>(getListOutputHandler( elemType, columnName, listWriterInterface ));
    return res;
}

PushInputAdapter *ParquetWriter::getStatusAdapter()
{
    return nullptr;
}

void ParquetWriter::start()
{
    std::vector<std::shared_ptr<arrow::Field>> arrowFields;
    if( !m_writeTimestampColumn.has_value() && !m_adapterMgr.getTimestampColumnName().empty() )
    {
        m_writeTimestampColumn = true;
        m_columnBuilders.push_back( std::make_shared<DatetimeArrayBuilder>( m_adapterMgr.getTimestampColumnName(), getChunkSize() ) );
        std::shared_ptr<arrow::KeyValueMetadata> colMetaData;
        auto colMetaIt = m_columnMetaData.find( m_adapterMgr.getTimestampColumnName() );
        if( colMetaIt != m_columnMetaData.end() )
        {
            colMetaData = colMetaIt -> second;
            m_columnMetaData.erase( colMetaIt );
        }

        arrowFields.push_back(
            arrow::field( m_adapterMgr.getTimestampColumnName(), m_columnBuilders.back() -> getDataType(), colMetaData ) );
    }
    else
    {
        m_writeTimestampColumn = false;
    }
    for( auto &&adapter:m_adapters )
    {
        for( unsigned i = 0; i < adapter -> getNumColumns(); ++i )
        {
            m_columnBuilders.push_back( adapter -> getColumnArrayBuilder( i ) );

            std::shared_ptr<arrow::KeyValueMetadata> colMetaData;
            auto colMetaIt = m_columnMetaData.find( m_columnBuilders.back() -> getColumnName() );
            if( colMetaIt != m_columnMetaData.end() )
            {
                colMetaData = colMetaIt -> second;
                m_columnMetaData.erase( colMetaIt );
            }
            arrowFields.push_back( arrow::field( m_columnBuilders.back() -> getColumnName(), 
                                                 m_columnBuilders.back() -> getDataType(), 
                                                 colMetaData ) );
        }
    }

    if( !m_columnMetaData.empty() )
        CSP_THROW( ValueError, "parquet column metadata has unmapped column: '" << m_columnMetaData.begin() -> first << "'" );
    initFileWriterContainer( arrow::schema( arrowFields, m_fileMetaData ) );
}

void ParquetWriter::stop()
{
    if( m_fileWriterWrapperContainer )
    {
        if( m_curChunkSize > 0 )
        {
            writeCurChunkToFile();
        }
        m_fileWriterWrapperContainer -> close();
        m_fileWriterWrapperContainer = nullptr;
    }
}

void ParquetWriter::onEndCycle()
{
    if( likely( isFileOpen() ) )
    {
        // This must be defined outside of the "if" below, the datetime object must live till the end of cycle since
        // We pass all values by reference
        DateTime now;
        if( m_writeTimestampColumn.value() )
        {
            // Set the timestamp value it's always the first
            now = m_adapterMgr.rootEngine() -> now();
            static_cast<DatetimeArrayBuilder *>(m_columnBuilders[ 0 ].get()) -> setValue( now );
        }
        for( auto &&columnBuilder:m_columnBuilders )
        {
            columnBuilder -> handleRowFinished();
        }
        if( ++m_curChunkSize >= getChunkSize() )
        {
            writeCurChunkToFile();
        }
    }
}

bool ParquetWriter::isFileOpen() const
{
    return m_fileWriterWrapperContainer != nullptr && m_fileWriterWrapperContainer -> isOpen();
}

void ParquetWriter::onFileNameChange( const std::string &fileName )
{
    CSP_TRUE_OR_THROW_RUNTIME( m_fileWriterWrapperContainer, "Trying to set file name when file writer already closed" );
    writeCurChunkToFile();
    m_fileWriterWrapperContainer -> close();
    if( !fileName.empty() )
    {
        m_fileWriterWrapperContainer
                -> open( fileName, m_adapterMgr.getCompression(), m_adapterMgr.isAllowOverwrite() );
    }
}

SingleColumnParquetOutputHandler *ParquetWriter::createScalarOutputHandler( CspTypePtr type, const std::string &name )
{
    return m_engine -> createOwnedObject<SingleColumnParquetOutputAdapter>( *this, type, name );
}

ListColumnParquetOutputHandler *ParquetWriter::createListOutputHandler( CspTypePtr &elemType, const std::string &columnName,
                                                                        DialectGenericListWriterInterface::Ptr listWriterInterface )
{
    return m_engine -> createOwnedObject<ListColumnParquetOutputAdapter>( *this, elemType, columnName, listWriterInterface );
}


StructParquetOutputHandler *ParquetWriter::createStructOutputHandler( CspTypePtr type,
                                                                      const DictionaryPtr &fieldMap )
{
    return m_engine -> createOwnedObject<StructParquetOutputAdapter>( *this, type, fieldMap );
}

void ParquetWriter::initFileWriterContainer( std::shared_ptr<arrow::Schema> schema )
{
    if( m_adapterMgr.isSplitColumnsToFiles() )
    {
        m_fileWriterWrapperContainer = std::make_unique<MultipleFileWriterWrapperContainer>( schema,
                                                                                             m_adapterMgr.isWriteArrowBinary() );
    }
    else
    {
        m_fileWriterWrapperContainer = std::make_unique<SingleFileWriterWrapperContainer>( schema,
                                                                                           m_adapterMgr.isWriteArrowBinary() );
    }
    if( !m_adapterMgr.getFileName().empty() )
    {
        m_fileWriterWrapperContainer -> open( m_adapterMgr.getFileName(),
                                              m_adapterMgr.getCompression(), m_adapterMgr.isAllowOverwrite() );
    }
}

void ParquetWriter::writeCurChunkToFile()
{
    if( m_curChunkSize > 0 )
    {
        if( unlikely( !isFileOpen() ) )
        {
            if(m_curChunkSize != 0 )
            {
                CSP_THROW( csp::RuntimeException, "Trying to write to parquet/arrow file, when no file name was provided" );

            }
            return;
        }
        m_fileWriterWrapperContainer -> writeData( m_columnBuilders );
        m_curChunkSize = 0;
    }
}

}
