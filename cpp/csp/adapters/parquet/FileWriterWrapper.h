#ifndef _IN_CSP_ADAPTERS_PARQUET_FileWriterWrapper_H
#define _IN_CSP_ADAPTERS_PARQUET_FileWriterWrapper_H

#include <unordered_map>
#include <string>
#include <memory>
#include <arrow/util/compression.h>

namespace arrow
{
class Schema;

class Table;
}

namespace csp::adapters::parquet
{

class FileWriterWrapper
{
public:
    FileWriterWrapper( std::shared_ptr<::arrow::Schema> schema )
            : m_schema( schema )
    {
    }

    FileWriterWrapper( const FileWriterWrapper &other ) = delete;
    FileWriterWrapper &operator=( const FileWriterWrapper &other ) = delete;

    virtual ~FileWriterWrapper() {}

    void open( const std::string &fileName, const std::string &compression, bool allowOverwrite = false );
    virtual void close() = 0;
    virtual void writeTable( const std::shared_ptr<::arrow::Table> &table ) = 0;

    const std::shared_ptr<::arrow::Schema> &getSchema() const { return m_schema; }

protected:
    virtual void openImpl( const std::string &fileName, const std::string &compression ) = 0;


    static ::arrow::Compression::type resolveCompression( const std::string &compression );
    static std::unique_ptr<arrow::util::Codec> resolveCompressionCodec( const std::string &compression );

private:
    std::shared_ptr<::arrow::Schema>                                   m_schema;
    static std::unordered_map<std::string, ::arrow::Compression::type> m_compressionNameMapping;

};

}


#endif
