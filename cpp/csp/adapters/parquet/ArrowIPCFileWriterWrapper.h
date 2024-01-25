#ifndef _IN_CSP_ADAPTERS_PARQUET_ArrowIPCFileWriterWrapper_H
#define _IN_CSP_ADAPTERS_PARQUET_ArrowIPCFileWriterWrapper_H

#include <csp/adapters/parquet/FileWriterWrapper.h>
#include <string>

namespace arrow::io
{
class FileOutputStream;
}

namespace arrow::ipc
{
class RecordBatchWriter;
}

namespace csp::adapters::parquet
{

class ArrowIPCFileWriterWrapper : public FileWriterWrapper
{
public:
    ArrowIPCFileWriterWrapper( std::shared_ptr<::arrow::Schema> schema )
            : FileWriterWrapper( schema )
    {
    }

    ~ArrowIPCFileWriterWrapper() { close(); }

    void close() override;
    void writeTable( const std::shared_ptr<::arrow::Table> &table ) override;

protected:
    void openImpl( const std::string &fileName, const std::string &compression ) override;

private:
    std::shared_ptr<::arrow::io::FileOutputStream> m_outputStream;
    std::shared_ptr<::arrow::ipc::RecordBatchWriter> m_fileWriter;
};

}


#endif