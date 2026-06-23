#include <csp/adapters/parquet/ParquetWriter.h>
#include <csp/adapters/parquet/ArrowBackedArrayBuilder.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/util/key_value_metadata.h>
#include <exception>

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
        m_fileMetaData = std::make_shared<::arrow::KeyValueMetadata>();
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

            auto kv_metadata = std::make_shared<::arrow::KeyValueMetadata>();

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

ListColumnParquetOutputHandler *ParquetWriter::getListOutputHandler( CspTypePtr &elemType, const std::string &columnName )
{
    CSP_TRUE_OR_THROW_RUNTIME( m_publishedColumnNames.emplace( columnName ).second,
                               "Trying to publish column " << columnName << " more than once" );
    //create and register adapter
    auto adapter = createListOutputHandler( elemType, columnName );
    m_adapters.emplace_back( adapter );
    return adapter;
}

ListColumnParquetOutputAdapter *ParquetWriter::getListOutputAdapter(
        CspTypePtr &elemType, const std::string &columnName )
{
    auto res = static_cast<ListColumnParquetOutputAdapter *>(getListOutputHandler( elemType, columnName ));
    return res;
}

void ParquetWriter::start()
{
    std::vector<std::shared_ptr<::arrow::Field>> arrowFields;
    // Honor an explicitly-set writeTimestampColumn; otherwise auto-detect from whether
    // a timestamp column name was configured.
    bool wantTimestamp = m_writeTimestampColumn.has_value()
                             ? m_writeTimestampColumn.value()
                             : !m_adapterMgr.getTimestampColumnName().empty();
    if( wantTimestamp )
    {
        CSP_TRUE_OR_THROW_RUNTIME( !m_adapterMgr.getTimestampColumnName().empty(),
            "writeTimestampColumn is true but no timestamp_column_name was provided" );
        m_writeTimestampColumn = true;
        auto tsBuilder = createArrowBackedArrayBuilder(
            m_adapterMgr.getTimestampColumnName(), getChunkSize(), CspType::DATETIME() );
        m_timestampBuilder = tsBuilder.get();
        m_columnBuilders.push_back( std::move( tsBuilder ) );
        std::shared_ptr<::arrow::KeyValueMetadata> colMetaData;
        auto colMetaIt = m_columnMetaData.find( m_adapterMgr.getTimestampColumnName() );
        if( colMetaIt != m_columnMetaData.end() )
        {
            colMetaData = colMetaIt -> second;
            m_columnMetaData.erase( colMetaIt );
        }

        arrowFields.push_back(
            ::arrow::field( m_adapterMgr.getTimestampColumnName(), m_columnBuilders.back() -> getDataType(), colMetaData ) );
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

            std::shared_ptr<::arrow::KeyValueMetadata> colMetaData;
            auto colMetaIt = m_columnMetaData.find( m_columnBuilders.back() -> getColumnName() );
            if( colMetaIt != m_columnMetaData.end() )
            {
                colMetaData = colMetaIt -> second;
                m_columnMetaData.erase( colMetaIt );
            }
            arrowFields.push_back( ::arrow::field( m_columnBuilders.back() -> getColumnName(), 
                                                 m_columnBuilders.back() -> getDataType(), 
                                                 colMetaData ) );
        }
    }

    if( !m_columnMetaData.empty() )
        CSP_THROW( ValueError, "parquet column metadata has unmapped column: '" << m_columnMetaData.begin() -> first << "'" );

    m_schema = ::arrow::schema( arrowFields, m_fileMetaData );
    // Hand the schema to the sink, then open the first file (if any). m_fileOpen reflects whether a
    // file is actually open after onFileChange -- an empty filename (no file / paused) leaves it false.
    const std::string & fileName = m_adapterMgr.getFileName();
    if( m_sink )
    {
        m_sink -> onStart( m_schema );
        m_fileOpen = m_sink -> onFileChange( fileName );
    }
    else
    {
        m_fileOpen = !fileName.empty();
    }
}

void ParquetWriter::stop()
{
    // Exception-safe teardown: a failure flushing the final batch must not skip onStop() (which
    // writes the file footer/closes the stream), and a failure in onStop() must not be masked.
    // Attempt both and rethrow the first error.
    std::exception_ptr firstError;
    if( m_curChunkSize > 0 )
    {
        try { flushBatch(); }
        catch( ... ) { firstError = std::current_exception(); }
    }
    if( m_sink )
    {
        try { m_sink -> onStop(); }
        catch( ... ) { if( !firstError ) firstError = std::current_exception(); }
    }
    m_fileOpen = false;
    if( firstError )
        std::rethrow_exception( firstError );
}

void ParquetWriter::adoptMetadataFrom( ParquetWriter & source )
{
    // Used by dict-basket writers: take the main writer's file-level metadata, and move any column
    // metadata the user attached to one of THIS writer's columns out of the source writer (so the
    // basket files carry it and the main writer no longer rejects those keys as "unmapped").
    if( !m_fileMetaData )
        m_fileMetaData = source.m_fileMetaData;
    for( auto && columnName : m_publishedColumnNames )
    {
        auto it = source.m_columnMetaData.find( columnName );
        if( it != source.m_columnMetaData.end() )
        {
            m_columnMetaData[ columnName ] = it -> second;
            source.m_columnMetaData.erase( it );
        }
    }
}

void ParquetWriter::onEndCycle()
{
    // When no file is open -- e.g. a filename_provider ticked "" to "pause" output -- this body is
    // skipped, so handleRowFinished() does not run for that cycle. As a result, a value that ticked
    // while paused is emitted on the first row after the file reopens, rather than being nulled.
    if( isFileOpen() ) [[likely]]
    {
        DateTime now;
        if( m_writeTimestampColumn.value() )
        {
            now = m_adapterMgr.rootEngine() -> now();
            m_timestampBuilder -> scratchField() -> setValue<DateTime>( m_timestampBuilder -> scratch(), now );
        }
        for( auto &&columnBuilder:m_columnBuilders )
        {
            columnBuilder -> handleRowFinished();
        }
        if( ++m_curChunkSize >= getChunkSize() )
        {
            flushBatch();
        }
    }
}

void ParquetWriter::onFileNameChange( const std::string &fileName )
{
    flushBatch();
    if( m_sink )
        m_fileOpen = m_sink -> onFileChange( fileName );
    else
        m_fileOpen = !fileName.empty();
}

SingleColumnParquetOutputHandler *ParquetWriter::createScalarOutputHandler( CspTypePtr type, const std::string &name )
{
    return m_engine -> createOwnedObject<SingleColumnParquetOutputAdapter>( *this, type, name );
}

ListColumnParquetOutputHandler *ParquetWriter::createListOutputHandler( CspTypePtr &elemType, const std::string &columnName )
{
    return m_engine -> createOwnedObject<ListColumnParquetOutputAdapter>( *this, elemType, columnName );
}


StructParquetOutputHandler *ParquetWriter::createStructOutputHandler( CspTypePtr type,
                                                                      const DictionaryPtr &fieldMap )
{
    return m_engine -> createOwnedObject<StructParquetOutputAdapter>( *this, type, fieldMap );
}

std::shared_ptr<::arrow::RecordBatch> ParquetWriter::buildRecordBatch()
{
    std::vector<std::shared_ptr<::arrow::Array>> columns;
    columns.reserve( m_columnBuilders.size() );
    for( auto && builder : m_columnBuilders )
    {
        auto column = builder -> buildArray();
        // Every column must contain exactly the rows accumulated this chunk; a mismatch would build a
        // corrupt RecordBatch. Enforced at runtime (not just in debug) so a future column-builder bug
        // can never silently produce a malformed parquet/arrow file.
        CSP_TRUE_OR_THROW_RUNTIME( column -> length() == static_cast<std::int64_t>( m_curChunkSize ),
            "ParquetWriter column '" << builder -> getColumnName() << "' produced " << column -> length()
            << " rows but the current chunk has " << m_curChunkSize );
        columns.push_back( std::move( column ) );
    }
    return ::arrow::RecordBatch::Make( m_schema, m_curChunkSize, std::move( columns ) );
}

void ParquetWriter::flushBatch()
{
    if( m_curChunkSize > 0 )
    {
        if( !isFileOpen() ) [[unlikely]]
            CSP_THROW( csp::RuntimeException, "Trying to write to parquet/arrow file, when no file name was provided" );
        if( m_sink )
            m_sink -> onBatch( buildRecordBatch() );
        m_curChunkSize = 0;
    }
}

}
