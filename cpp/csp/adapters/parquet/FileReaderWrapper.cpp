#include <csp/adapters/parquet/FileReaderWrapper.h>
#include <arrow/io/file.h>
#include <parquet/api/io.h>


namespace csp::adapters::parquet
{

FileReaderWrapper::~FileReaderWrapper()
{
    close();
}

void FileReaderWrapper::open( const std::string &fileName )
{
    if( m_inputFile )
    {
        close();
    }
    PARQUET_ASSIGN_OR_THROW(
            m_inputFile,
            arrow::io::ReadableFile::Open( fileName ) );
    m_fileName = fileName;
}

void FileReaderWrapper::close()
{
    m_fileName.clear();
    if( m_inputFile )
    {
        m_inputFile = nullptr;
    }
}

}