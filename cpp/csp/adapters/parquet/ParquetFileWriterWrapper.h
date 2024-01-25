#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetFileWriterWrapper_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetFileWriterWrapper_H

#include <csp/adapters/parquet/FileWriterWrapper.h>
#include <string>

namespace arrow::io
{
class FileOutputStream;
}
namespace parquet::arrow
{
class FileWriter;
}

namespace csp::adapters::parquet
{

class ParquetFileWriterWrapper : public FileWriterWrapper
{
public:
    ParquetFileWriterWrapper( std::shared_ptr<::arrow::Schema> schema )
            : FileWriterWrapper( schema )
    {
    }

    ~ParquetFileWriterWrapper() { close(); }

    void close() override;
    void writeTable( const std::shared_ptr<::arrow::Table> &table ) override;

protected:
    void openImpl( const std::string &fileName, const std::string &compression ) override;

private:
    std::shared_ptr<::arrow::io::FileOutputStream> m_outputStream;
    std::unique_ptr<::parquet::arrow::FileWriter>  m_fileWriter;
};

}


#endif