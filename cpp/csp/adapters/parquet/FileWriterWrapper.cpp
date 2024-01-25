#include <csp/adapters/parquet/FileWriterWrapper.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/Exception.h>
#include <csp/core/FileUtils.h>
#include <string>
#include <filesystem>

namespace csp::adapters::parquet
{
std::unordered_map<std::string, ::arrow::Compression::type> FileWriterWrapper::m_compressionNameMapping{
        { "",          ::arrow::Compression::UNCOMPRESSED },
        { "snappy",    ::arrow::Compression::SNAPPY },
        { "gzip",      ::arrow::Compression::GZIP },
        { "brotli",    ::arrow::Compression::BROTLI },
        { "zstd",      ::arrow::Compression::ZSTD },
        { "lz4",       ::arrow::Compression::LZ4 },
        { "lz4_frame", ::arrow::Compression::LZ4_FRAME },
        { "lzo",       ::arrow::Compression::LZO },
        { "bz2",       ::arrow::Compression::BZ2 }
};

void FileWriterWrapper::open( const std::string &fileName, const std::string &compression, bool allowOverwrite )
{
    if( !allowOverwrite )
    {
        CSP_TRUE_OR_THROW_RUNTIME( !utils::fileExists( fileName ),
                                   "Trying to overwrite existing file " << fileName << " while allow_overwrite is false" );
    }

    utils::mkdir( utils::dirname( fileName ) );
    openImpl( fileName, compression );
}

::arrow::Compression::type FileWriterWrapper::resolveCompression( const std::string &compression )
{
    auto it = m_compressionNameMapping.find( compression );
    CSP_TRUE_OR_THROW_RUNTIME( it != m_compressionNameMapping.end(), "Unable to resolve compression: " << compression );
    return it->second;
}

std::unique_ptr<arrow::util::Codec> FileWriterWrapper::resolveCompressionCodec( const std::string &compression )
{
    auto compressionType = resolveCompression( compression );
    auto res = arrow::util::Codec::Create( compressionType );
    STATUS_OK_OR_THROW_RUNTIME(res.status(), "Failed to create arrow codec for " << compressionType );
    return std::move(res.ValueUnsafe());
}

}
