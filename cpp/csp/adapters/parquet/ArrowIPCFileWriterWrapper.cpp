#include <csp/adapters/parquet/ArrowIPCFileWriterWrapper.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/ipc/writer.h>
#include <parquet/api/io.h>

namespace csp::adapters::parquet
{
void ArrowIPCFileWriterWrapper::openImpl( const std::string &fileName, const std::string &compression )
{
    CSP_TRUE_OR_THROW_RUNTIME( m_outputStream == nullptr, "Trying to open parquet file while previous was not closed" );

    PARQUET_ASSIGN_OR_THROW(
            m_outputStream,
            arrow::io::FileOutputStream::Open( fileName.c_str()));

    arrow::ipc::IpcWriteOptions writeOptions;
    writeOptions.codec  = resolveCompressionCodec( compression );

    STATUS_OK_OR_THROW_RUNTIME(
            ::arrow::ipc::MakeStreamWriter( m_outputStream.get(), getSchema(), writeOptions ).Value(&m_fileWriter),
            "Failed to open arrow file writer" );
}

void ArrowIPCFileWriterWrapper::close()
{
    if( m_outputStream )
    {
        // Let's move them first, if there are any exceptions, we still want the pointer to be null
        std::shared_ptr<::arrow::io::FileOutputStream> outputStream{ std::move( m_outputStream ) };
        std::shared_ptr<::arrow::ipc::RecordBatchWriter>  fileWriter{ std::move( m_fileWriter ) };
        // Should be done by move constructor but let's be safe:
        m_outputStream = nullptr;
        m_fileWriter   = nullptr;

        if( fileWriter )
            STATUS_OK_OR_THROW_RUNTIME( fileWriter->Close(), "Failed to close arrow file writer" );
        if( outputStream )
            STATUS_OK_OR_THROW_RUNTIME( outputStream->Close(), "Failed to close arrow output stream" );
    }
}

void ArrowIPCFileWriterWrapper::writeTable( const std::shared_ptr<::arrow::Table> &table )
{
    STATUS_OK_OR_THROW_RUNTIME( m_fileWriter->WriteTable( *table, table->num_rows()), "Failed to write to arrow file" );
}

}
