#include <csp/adapters/parquet/ParquetFileReaderWrapper.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <parquet/arrow/reader.h>
#include <arrow/io/file.h>
#include <arrow/util/config.h>

namespace csp::adapters::parquet
{
ParquetFileReaderWrapper::~ParquetFileReaderWrapper()
{
    close();
}

void ParquetFileReaderWrapper::open( const std::string &fileName )
{
    FileReaderWrapper::open( fileName );

    try
    {
#if ARROW_VERSION_MAJOR >= 20
        auto res = ::parquet::arrow::OpenFile(m_inputFile, arrow::default_memory_pool());
        STATUS_OK_OR_THROW_RUNTIME(res.status(), "Failed to open parquet file " << fileName );
        m_fileReader = res.MoveValueUnsafe();
#else
        STATUS_OK_OR_THROW_RUNTIME(
                ::parquet::arrow::OpenFile( m_inputFile, arrow::default_memory_pool(), &m_fileReader ),
                "Failed to open parquet file " << fileName );
#endif
    }
    catch( ... )
    {
        FileReaderWrapper::close();
        throw;
    }
    m_nextRowGroup = 0;
}

void ParquetFileReaderWrapper::close()
{
    m_fileReader = nullptr;
    FileReaderWrapper::close();
}

bool ParquetFileReaderWrapper::readNextRowGroup( const std::vector<int> neededColumns, std::shared_ptr<::arrow::Table> &dst )
{
    if( m_fileReader -> num_row_groups() > m_nextRowGroup )
    {
        STATUS_OK_OR_THROW_RUNTIME( m_fileReader -> ReadRowGroup( m_nextRowGroup, neededColumns, &dst ),
                                    "Failed to read row group " << m_nextRowGroup << " from file " << m_fileName );
        ++m_nextRowGroup;
        return true;
    }
    else
    {
        dst = nullptr;
        return false;

    }
}

void ParquetFileReaderWrapper::getSchema( std::shared_ptr<::arrow::Schema> &dst )
{
    STATUS_OK_OR_THROW_RUNTIME( m_fileReader -> GetSchema( &dst ),
                                "Failed to get schema from file " << m_fileName );
}

}