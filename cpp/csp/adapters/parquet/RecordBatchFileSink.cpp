#include <csp/adapters/parquet/RecordBatchFileSink.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/FileUtils.h>

#include <arrow/io/file.h>
#include <arrow/ipc/writer.h>
#include <arrow/record_batch.h>
#include <arrow/table.h>
#include <arrow/type.h>
#include <arrow/util/compression.h>
#include <arrow/util/config.h>
#include <parquet/arrow/writer.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace csp::adapters::parquet
{
namespace
{

// A single open output file: write one batch, then close. Type-erased over format
// so the rest of the machinery never branches on parquet-vs-ipc-vs-split.
struct FileWriter
{
    std::function<void( const std::shared_ptr<::arrow::RecordBatch> & )> writeBatch;
    std::function<void()>                                                close;
};

// (path, schema) -> open FileWriter.  The factory IS the format/layout choice.
using FileWriterFactory =
    std::function<FileWriter( const std::string &, const std::shared_ptr<::arrow::Schema> & )>;

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

FileWriter makeParquetFileWriter( const std::string & path, const std::shared_ptr<::arrow::Schema> & schema,
                                  const std::string & compression, bool allowOverwrite )
{
    auto stream = openOutputStream( path, allowOverwrite );

    ::parquet::WriterProperties::Builder props;
    props.compression( resolveCompression( compression ) );
#if ARROW_VERSION_MAJOR >= 20
    props.version( ::parquet::ParquetVersion::PARQUET_2_6 );
#else
    props.version( ::parquet::ParquetVersion::PARQUET_2_0 );
#endif
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

FileWriter makeIpcFileWriter( const std::string & path, const std::shared_ptr<::arrow::Schema> & schema,
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
        [writer]( const std::shared_ptr<::arrow::RecordBatch> & rb )
        {
            STATUS_OK_OR_THROW_RUNTIME( writer -> WriteRecordBatch( *rb ), "Failed to write arrow record batch" );
        },
        [writer, stream]()
        {
            auto writerStatus = writer -> Close();
            auto streamStatus = stream -> Close();   // always close the stream, even if the writer close failed
            STATUS_OK_OR_THROW_RUNTIME( writerStatus, "Failed to close arrow IPC writer" );
            STATUS_OK_OR_THROW_RUNTIME( streamStatus, "Failed to close arrow output stream" );
        } };
}

// A split-columns writer is itself a FileWriter that fans out to one sub-writer per
// column.  Each column file carries the parent file-level metadata.
FileWriter makeSplitWriter( const std::string & dir, const std::shared_ptr<::arrow::Schema> & schema,
                            const FileWriterFactory & perColumn, const std::string & extension )
{
    if( !dir.empty() )
        utils::mkdir( dir );

    auto subWriters = std::make_shared<std::vector<FileWriter>>();
    auto colSchemas = std::make_shared<std::vector<std::shared_ptr<::arrow::Schema>>>();
    subWriters -> reserve( schema -> num_fields() );
    colSchemas -> reserve( schema -> num_fields() );

    for( int i = 0; i < schema -> num_fields(); ++i )
    {
        auto colSchema = ::arrow::schema( { schema -> field( i ) }, schema -> metadata() );
        colSchemas -> push_back( colSchema );
        subWriters -> push_back( perColumn( dir + "/" + schema -> field( i ) -> name() + extension, colSchema ) );
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

} // namespace

RecordBatchSink makeFileSink( bool writeArrowBinary, bool splitColumns,
                              const std::string & compression, bool allowOverwrite,
                              std::function<void( const std::string & )> fileVisitor )
{
    const std::string extension = writeArrowBinary ? ".arrow" : ".parquet";

    // Per-file factory: format choice is the only difference.
    FileWriterFactory perFile =
        [writeArrowBinary, compression, allowOverwrite]( const std::string & path, const std::shared_ptr<::arrow::Schema> & schema )
        {
            return writeArrowBinary ? makeIpcFileWriter( path, schema, compression, allowOverwrite )
                                    : makeParquetFileWriter( path, schema, compression, allowOverwrite );
        };

    // Layout choice: single file vs. one file per column.
    FileWriterFactory factory = perFile;
    if( splitColumns )
        factory = [perFile, extension]( const std::string & dir, const std::shared_ptr<::arrow::Schema> & schema )
        {
            return makeSplitWriter( dir, schema, perFile, extension );
        };

    // State shared across the sink callbacks.
    auto schemaHolder = std::make_shared<std::shared_ptr<::arrow::Schema>>();
    auto current      = std::make_shared<std::optional<FileWriter>>();
    auto currentPath  = std::make_shared<std::string>();

    auto closeCurrent = [current, currentPath, fileVisitor]()
    {
        if( current -> has_value() )
        {
            // Detach + reset BEFORE invoking the (user-supplied) visitor, so that a throwing
            // visitor cannot cause a later onStop()->closeCurrent() to close this writer twice.
            FileWriter w = std::move( current -> value() );
            current -> reset();
            w.close();
            if( fileVisitor )
                fileVisitor( *currentPath );
        }
    };

    RecordBatchSink sink;
    sink.onStart      = [schemaHolder]( const std::shared_ptr<::arrow::Schema> & schema ) { *schemaHolder = schema; };
    sink.onBatch      = [current]( const std::shared_ptr<::arrow::RecordBatch> & rb )
    {
        if( current -> has_value() )
            ( *current ) -> writeBatch( rb );
    };
    sink.onFileChange = [factory, schemaHolder, current, currentPath, closeCurrent]( const std::string & path )
    {
        closeCurrent();
        if( !path.empty() )
        {
            *current     = factory( path, *schemaHolder );
            *currentPath = path;
        }
    };
    sink.onStop = closeCurrent;
    return sink;
}

} // namespace csp::adapters::parquet
