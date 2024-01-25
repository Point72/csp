#include <csp/adapters/parquet/ArrowIPCFileReaderWrapper.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <parquet/exception.h>

namespace csp::adapters::parquet
{
ArrowIPCFileReaderWrapper::~ArrowIPCFileReaderWrapper()
{
    close();
}

void ArrowIPCFileReaderWrapper::open( const std::string &fileName )
{
    FileReaderWrapper::open( fileName );

    try
    {
        PARQUET_ASSIGN_OR_THROW( m_fileReader,
                                 ::arrow::ipc::RecordBatchStreamReader::Open( m_inputFile ) );
    }
    catch( std::exception &e )
    {
        FileReaderWrapper::close();
        CSP_THROW( RuntimeException, "Failed to open " << fileName << ":" << e.what() );
    }
}

void ArrowIPCFileReaderWrapper::close()
{
    m_fileReader = nullptr;
    FileReaderWrapper::close();
}

bool ArrowIPCFileReaderWrapper::readNextRowGroup( const std::vector<int> neededColumns, std::shared_ptr<::arrow::Table> &dst )
{
    std::shared_ptr<arrow::RecordBatch> recordBatch;
    STATUS_OK_OR_THROW_RUNTIME( m_fileReader -> ReadNext( &recordBatch ), "Failed to read next record batch" );
    if( recordBatch == nullptr )
    {
        dst = nullptr;
        return false;
    }
    std::vector<std::shared_ptr<arrow::Array>> tableColumns;
    std::vector<std::shared_ptr<arrow::Field>> tableColumnSpecs;
    tableColumns.reserve(neededColumns.size());
    tableColumnSpecs.reserve(neededColumns.size());

    for( auto &&columnIndex : neededColumns )
    {
        tableColumns.push_back(recordBatch->column(columnIndex));
        tableColumnSpecs.push_back(recordBatch->schema()->field(columnIndex));
    }
    dst = arrow::Table::Make( ::arrow::schema( tableColumnSpecs ), tableColumns );
    return true;
}


void ArrowIPCFileReaderWrapper::getSchema( std::shared_ptr<::arrow::Schema> &dst )
{
    dst = m_fileReader -> schema();
}

}