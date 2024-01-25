#ifndef _IN_CSP_ADAPTERS_PARQUET_FileWriterWrapperContainer_H
#define _IN_CSP_ADAPTERS_PARQUET_FileWriterWrapperContainer_H

#include <string>
#include <vector>
#include <memory>
#include <csp/adapters/parquet/FileWriterWrapper.h>

namespace arrow
{
class Schema;
}

namespace csp::adapters::parquet
{
class ArrowSingleColumnArrayBuilder;

class FileWriterWrapperContainer
{
public:
    FileWriterWrapperContainer() = default;
    FileWriterWrapperContainer( FileWriterWrapperContainer & ) = delete;
    FileWriterWrapperContainer &operator=( const FileWriterWrapperContainer & ) = delete;
    virtual ~FileWriterWrapperContainer() = default;

    virtual void open( const std::string &fileName, const std::string &compression, bool allowOverwrite = false ) = 0;

    bool isOpen() const{ return m_isOpen; }

    virtual void close() = 0;
    virtual void writeData( const std::vector<std::shared_ptr<ArrowSingleColumnArrayBuilder>> &columnBuilders ) = 0;
protected:
    void setOpen( bool open ){ m_isOpen = open; }

protected:
    using WriterPtr = std::unique_ptr<FileWriterWrapper>;

    static WriterPtr createSingleFileWrapper( const std::shared_ptr<arrow::Schema> &schema, bool isWriteArrowBinary );
private:
    bool m_isOpen = false;
};

class SingleFileWriterWrapperContainer final : public FileWriterWrapperContainer
{
public:
    SingleFileWriterWrapperContainer( std::shared_ptr<arrow::Schema> schema, bool isWriteArrowBinary );
    virtual void open( const std::string &fileName, const std::string &compression, bool allowOverwrite = false ) override;
    virtual void close() override;
    virtual void writeData( const std::vector<std::shared_ptr<ArrowSingleColumnArrayBuilder>> &columnBuilders ) override;
private:
    std::unique_ptr<FileWriterWrapper> m_fileWriterWrapper;
};

class MultipleFileWriterWrapperContainer : public FileWriterWrapperContainer
{
public:
    MultipleFileWriterWrapperContainer( std::shared_ptr<arrow::Schema> schema, bool isWriteArrowBinary );
    virtual void open( const std::string &fileName, const std::string &compression, bool allowOverwrite = false ) override;
    virtual void close() override;
    virtual void writeData( const std::vector<std::shared_ptr<ArrowSingleColumnArrayBuilder>> &columnBuilders ) override;
private:
    struct SingleFileWrapperInfo
    {
        std::string                        m_fileName;
        std::unique_ptr<FileWriterWrapper> m_fileWriterWrapper;
    };

private:
    std::vector<SingleFileWrapperInfo> m_fileWriterWrappers;
};

}

#endif
