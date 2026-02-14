#include <csp/adapters/parquet/ParquetFileWriterWrapper.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/util/config.h>
#include <parquet/arrow/writer.h>

namespace csp::adapters::parquet
{
void ParquetFileWriterWrapper::openImpl( const std::string &fileName, const std::string &compression )
{
    CSP_TRUE_OR_THROW_RUNTIME( m_outputStream == nullptr, "Trying to open parquet file while previous was not closed" );

    PARQUET_ASSIGN_OR_THROW(
            m_outputStream,
            arrow::io::FileOutputStream::Open( fileName.c_str()));

    ::parquet::WriterProperties::Builder builder;
    builder.compression( resolveCompression( compression ));
#if ARROW_VERSION_MAJOR >= 20
    builder.version(::parquet::ParquetVersion::PARQUET_2_6 );
#else
    builder.version(::parquet::ParquetVersion::PARQUET_2_0 );
#endif
    ::parquet::ArrowWriterProperties::Builder arrowBuilder;
    arrowBuilder.store_schema();

    auto res = ::parquet::arrow::FileWriter::Open( *getSchema(), arrow::default_memory_pool(), m_outputStream, builder.build(), arrowBuilder.build() );
    STATUS_OK_OR_THROW_RUNTIME( res.status(), "Failed to open parquet file writer" );
    m_fileWriter = res.MoveValueUnsafe();
}

void ParquetFileWriterWrapper::close()
{
    if( m_outputStream )
    {
        // Let's move them first, if there are any exceptions, we still want the pointer to be null
        std::shared_ptr<::arrow::io::FileOutputStream> outputStream{ std::move( m_outputStream ) };
        std::unique_ptr<::parquet::arrow::FileWriter>  fileWriter{ std::move( m_fileWriter ) };
        // Should be done by move constructor but let's be safe:
        m_outputStream = nullptr;
        m_fileWriter   = nullptr;

        if(fileWriter)
            STATUS_OK_OR_THROW_RUNTIME( fileWriter->Close(), "Failed to close parquet file writer" );
        if(outputStream)
            STATUS_OK_OR_THROW_RUNTIME( outputStream->Close(), "Failed to close parquet output stream" );
    }
}

void ParquetFileWriterWrapper::writeTable( const std::shared_ptr<::arrow::Table> &table )
{
    CSP_TRUE_OR_THROW_RUNTIME(m_fileWriter, "File writer is null!!!");
    STATUS_OK_OR_THROW_RUNTIME( m_fileWriter->WriteTable( *table, table->num_rows()), "Failed to write to parquet file" );
}

}
