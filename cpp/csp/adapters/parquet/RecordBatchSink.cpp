#include <csp/adapters/parquet/RecordBatchSink.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/FileUtils.h>

#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/compression.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace csp::adapters::parquet
{

namespace
{

::arrow::Compression::type resolveCompression( const std::string & name )
{
    // csp treats "" / "none" as no compression; Arrow expects lower-case codec names.
    std::string lower( name );
    std::transform( lower.begin(), lower.end(), lower.begin(),
                    []( unsigned char c ) { return static_cast<char>( std::tolower( c ) ); } );
    if( lower.empty() || lower == "none" )
        return ::arrow::Compression::UNCOMPRESSED;

    // Arrow is the source of truth for which compression names exist...
    auto compressionType = ::arrow::util::Codec::GetCompressionType( lower );
    CSP_TRUE_OR_THROW_RUNTIME( compressionType.ok(), "Unsupported compression '" << name << "'" );
    // ...and whether this Arrow build was actually compiled with support for it.
    CSP_TRUE_OR_THROW_RUNTIME( ::arrow::util::Codec::IsAvailable( compressionType.ValueUnsafe() ),
        "Compression '" << name << "' is not available in this Arrow build" );
    return compressionType.ValueUnsafe();
}

// Enforce overwrite policy, create parent dirs, and open the output stream.
std::shared_ptr<::arrow::io::OutputStream> openOutputStream( const std::string & path, bool allowOverwrite )
{
    if( !allowOverwrite && utils::fileExists( path ) )
        CSP_THROW( FileExistsError, "Trying to overwrite existing file " << path << " while allow_overwrite is false" );
    // Only create the parent directory when the path has one; a bare relative
    // filename (e.g. "out.parquet") has an empty dirname, and mkdir("") fails.
    auto parentDir = utils::dirname( path );
    if( !parentDir.empty() )
        utils::mkdir( parentDir );

    auto result = ::arrow::io::FileOutputStream::Open( path );
    STATUS_OK_OR_THROW_RUNTIME( result.status(), "Failed to open output stream " << path );
    return result.MoveValueUnsafe();
}

} // namespace

RecordBatchSink::FileWriter RecordBatchSink::makeParquetFileWriter(
    const std::string & path, const std::shared_ptr<::arrow::Schema> & schema,
    const std::string & compression, bool allowOverwrite, bool useDictionary )
{
    auto stream = openOutputStream( path, allowOverwrite );

    ::parquet::WriterProperties::Builder props;
    props.compression( resolveCompression( compression ) );
    // csp builds against libparquet >= 16.0.0, where PARQUET_2_6 (the modern default, available since
    // Arrow 6.0) is always present, so there is no need to gate on the Arrow version or fall back to
    // the deprecated PARQUET_2_0.
    props.version( ::parquet::ParquetVersion::PARQUET_2_6 );
    // Dictionary encoding is on by default. For high-cardinality numeric columns it is a large write-time
    // cost (hash table build) and can even grow files; allow it to be turned off (trading file size for
    // write throughput). Low-cardinality columns still benefit from leaving it on (the default).
    if( !useDictionary )
        props.disable_dictionary();
    ::parquet::ArrowWriterProperties::Builder arrowProps;
    arrowProps.store_schema(); // preserve arrow (file/column) schema metadata in the parquet file

    auto result = ::parquet::arrow::FileWriter::Open(
        *schema, ::arrow::default_memory_pool(), stream, props.build(), arrowProps.build() );
    STATUS_OK_OR_THROW_RUNTIME( result.status(), "Failed to open parquet writer for " << path );
    std::shared_ptr<::parquet::arrow::FileWriter> writer = result.MoveValueUnsafe();

    return FileWriter{
        [writer]( const std::shared_ptr<::arrow::RecordBatch> & rb )
        {
            // Write each flushed batch as its own row group (chunk_size = num_rows),
            // matching the batch_size -> row-group contract.
            auto table = ::arrow::Table::FromRecordBatches( rb -> schema(), { rb } );
            STATUS_OK_OR_THROW_RUNTIME( table.status(), "Failed to assemble table for parquet write" );
            STATUS_OK_OR_THROW_RUNTIME( writer -> WriteTable( *table.ValueUnsafe(), rb -> num_rows() ),
                                        "Failed to write parquet record batch" );
        },
        [writer, stream]()
        {
            auto writerStatus = writer -> Close();
            auto streamStatus = stream -> Close();   // always close the stream, even if the footer flush failed
            STATUS_OK_OR_THROW_RUNTIME( writerStatus, "Failed to close parquet writer" );
            STATUS_OK_OR_THROW_RUNTIME( streamStatus, "Failed to close parquet output stream" );
        } };
}

RecordBatchSink::FileWriter RecordBatchSink::makeIpcFileWriter(
    const std::string & path, const std::shared_ptr<::arrow::Schema> & schema,
    const std::string & compression, bool allowOverwrite )
{
    auto stream = openOutputStream( path, allowOverwrite );

    auto options         = ::arrow::ipc::IpcWriteOptions::Defaults();
    auto compressionType = resolveCompression( compression );
    if( compressionType != ::arrow::Compression::UNCOMPRESSED )
    {
        auto codec = ::arrow::util::Codec::Create( compressionType );
        STATUS_OK_OR_THROW_RUNTIME( codec.status(), "Failed to create codec for " << compression );
        options.codec = codec.MoveValueUnsafe();
    }

    auto result = ::arrow::ipc::MakeStreamWriter( stream, schema, options );
    STATUS_OK_OR_THROW_RUNTIME( result.status(), "Failed to open arrow IPC writer for " << path );
    std::shared_ptr<::arrow::ipc::RecordBatchWriter> writer = result.MoveValueUnsafe();

    return FileWriter{
        [writer, compression]( const std::shared_ptr<::arrow::RecordBatch> & rb )
        {
            // Arrow validates which codecs IPC allows here and appends its own reason (e.g. "Only
            // LZ4_FRAME and ZSTD compression allowed for IPC"); surface which codec was requested
            // rather than a bare "failed to write". The allowed set is Arrow's to define, not ours.
            STATUS_OK_OR_THROW_RUNTIME( writer -> WriteRecordBatch( *rb ),
                "Failed to write Arrow IPC record batch with compression '" << compression << "'" );
        },
        [writer, stream]()
        {
            auto writerStatus = writer -> Close();
            auto streamStatus = stream -> Close();   // always close the stream, even if the writer close failed
            STATUS_OK_OR_THROW_RUNTIME( writerStatus, "Failed to close arrow IPC writer" );
            STATUS_OK_OR_THROW_RUNTIME( streamStatus, "Failed to close arrow output stream" );
        } };
}

// A split-columns writer is itself a FileWriter that fans out to one sub-writer per column.
// Each column file carries the parent file-level metadata.
RecordBatchSink::FileWriter RecordBatchSink::makeSplitWriter(
    const std::string & dir, const std::shared_ptr<::arrow::Schema> & schema,
    bool writeArrowBinary, const std::string & compression, bool allowOverwrite,
    bool useDictionary, const std::string & extension )
{
    if( !dir.empty() )
        utils::mkdir( dir );

    auto subWriters = std::make_shared<std::vector<FileWriter>>();
    auto colSchemas = std::make_shared<std::vector<std::shared_ptr<::arrow::Schema>>>();
    subWriters -> reserve( schema -> num_fields() );
    colSchemas -> reserve( schema -> num_fields() );

    try
    {
        for( int i = 0; i < schema -> num_fields(); ++i )
        {
            auto colSchema = ::arrow::schema( { schema -> field( i ) }, schema -> metadata() );
            colSchemas -> push_back( colSchema );
            auto colPath = dir + "/" + schema -> field( i ) -> name() + extension;
            subWriters -> push_back( writeArrowBinary
                ? makeIpcFileWriter( colPath, colSchema, compression, allowOverwrite )
                : makeParquetFileWriter( colPath, colSchema, compression, allowOverwrite, useDictionary ) );
        }
    }
    catch( ... )
    {
        // A later column failed to open (e.g. allow_overwrite=false and that file already exists).
        // Close the sub-writers already opened so we don't leak file descriptors or leave half-open
        // files behind, then rethrow the original error.
        for( auto & w : *subWriters )
        {
            try { w.close(); }
            catch( ... ) {}
        }
        throw;
    }

    return FileWriter{
        [subWriters, colSchemas]( const std::shared_ptr<::arrow::RecordBatch> & rb )
        {
            for( int i = 0; i < rb -> num_columns(); ++i )
            {
                auto colBatch = ::arrow::RecordBatch::Make( ( *colSchemas )[i], rb -> num_rows(), { rb -> column( i ) } );
                ( *subWriters )[i].writeBatch( colBatch );
            }
        },
        [subWriters]()
        {
            std::exception_ptr firstError;
            for( auto & w : *subWriters )
            {
                try { w.close(); }
                catch( ... ) { if( !firstError ) firstError = std::current_exception(); }
            }
            if( firstError )
                std::rethrow_exception( firstError );
        } };
}

RecordBatchSink::RecordBatchSink( bool writeArrowBinary, bool splitColumns,
                                  std::string compression, bool allowOverwrite,
                                  std::function<void( const std::string & )> fileVisitor,
                                  bool useDictionary )
    : m_writeArrowBinary( writeArrowBinary )
    , m_splitColumns( splitColumns )
    , m_compression( std::move( compression ) )
    , m_allowOverwrite( allowOverwrite )
    , m_fileVisitor( std::move( fileVisitor ) )
    , m_useDictionary( useDictionary )
    , m_extension( writeArrowBinary ? ".arrow" : ".parquet" )
{
}


RecordBatchSink::FileWriter RecordBatchSink::openWriter( const std::string & path )
{
    if( m_splitColumns )
        return makeSplitWriter( path, m_schema, m_writeArrowBinary, m_compression, m_allowOverwrite,
                                m_useDictionary, m_extension );
    return m_writeArrowBinary ? makeIpcFileWriter( path, m_schema, m_compression, m_allowOverwrite )
                              : makeParquetFileWriter( path, m_schema, m_compression, m_allowOverwrite, m_useDictionary );
}

void RecordBatchSink::closeCurrent()
{
    if( m_current.has_value() )
    {
        // Detach + reset BEFORE invoking the (user-supplied) visitor, so that a throwing visitor
        // cannot cause a later onStop()->closeCurrent() to close this writer twice.
        FileWriter w = std::move( *m_current );
        m_current.reset();
        w.close();
        if( m_fileVisitor )
            m_fileVisitor( m_currentPath );
    }
}

void RecordBatchSink::onStart( const std::shared_ptr<::arrow::Schema> & schema )
{
    m_schema = schema;
}

void RecordBatchSink::onBatch( const std::shared_ptr<::arrow::RecordBatch> & rb )
{
    // A batch must only ever arrive while a file is open (the writer guards this via isFileOpen()
    // before flushing). If the writer and sink ever desync, fail loudly instead of silently
    // dropping the batch's rows.
    CSP_TRUE_OR_THROW_RUNTIME( m_current.has_value(),
        "RecordBatchSink received a record batch with no open output file" );
    m_current -> writeBatch( rb );
}

bool RecordBatchSink::onFileChange( const std::string & path )
{
    closeCurrent();
    if( path.empty() )
        return false;
    m_current     = openWriter( path );
    m_currentPath = path;
    return true;
}

void RecordBatchSink::onStop()
{
    closeCurrent();
}

} // namespace csp::adapters::parquet
