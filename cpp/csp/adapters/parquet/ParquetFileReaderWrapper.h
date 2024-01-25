#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetFileReaderWrapper_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetFileReaderWrapper_H

#include <csp/adapters/parquet//FileReaderWrapper.h>
#include <string>

namespace parquet::arrow
{
class FileReader;
}

namespace csp::adapters::parquet
{

class ParquetFileReaderWrapper : public FileReaderWrapper
{
public:
    ~ParquetFileReaderWrapper();
    virtual void open( const std::string &fileName ) override;
    virtual void close() override;

    virtual bool readNextRowGroup( const std::vector<int> neededColumns, std::shared_ptr<::arrow::Table> &dst ) override;
    void getSchema( std::shared_ptr<::arrow::Schema> &dst ) override;

private:
    std::unique_ptr<::parquet::arrow::FileReader> m_fileReader;
    int                                           m_nextRowGroup;
};
}

#endif
