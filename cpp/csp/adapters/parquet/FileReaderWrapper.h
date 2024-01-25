#ifndef _IN_CSP_ADAPTERS_PARQUET_FileReaderWrapper_H
#define _IN_CSP_ADAPTERS_PARQUET_FileReaderWrapper_H

#include <memory>
#include <string>
#include <vector>

namespace arrow::io
{
class ReadableFile;
}

namespace arrow
{
class Schema;
class Table;
}

namespace csp::adapters::parquet
{

class FileReaderWrapper
{
public:
    using Ptr = std::unique_ptr<FileReaderWrapper>;

    FileReaderWrapper() = default;
    FileReaderWrapper( const FileReaderWrapper & ) = delete;
    FileReaderWrapper &operator=( const FileReaderWrapper & ) = delete;
    virtual ~FileReaderWrapper();

    virtual void open( const std::string &fileName );
    virtual void close();

    virtual bool readNextRowGroup( const std::vector<int> neededColumns, std::shared_ptr<::arrow::Table> &dst ) = 0;
    virtual void getSchema(std::shared_ptr<::arrow::Schema>& dst) = 0;

    const std::string &getCurrentFileName() const{ return m_fileName; }

protected:
    std::shared_ptr<::arrow::io::ReadableFile> m_inputFile;

protected:
    std::string m_fileName;
};

using FileReaderWrapperPtr = FileReaderWrapper::Ptr;

}
#endif
