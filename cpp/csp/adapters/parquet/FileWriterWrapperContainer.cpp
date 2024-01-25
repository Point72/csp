#include <csp/adapters/parquet/FileWriterWrapperContainer.h>
#include <csp/adapters/parquet/ParquetWriter.h>
#include <csp/adapters/parquet/ArrowIPCFileWriterWrapper.h>
#include <csp/adapters/parquet/ParquetFileWriterWrapper.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <arrow/array.h>
#include <arrow/builder.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>
#include <csp/core/Exception.h>

namespace csp::adapters::parquet
{

FileWriterWrapperContainer::WriterPtr FileWriterWrapperContainer::createSingleFileWrapper( const std::shared_ptr<arrow::Schema> &schema,
                                                                                           bool isWriteArrowBinary )
{
    if( isWriteArrowBinary )
    {
        return std::make_unique<ArrowIPCFileWriterWrapper>( schema );
    }
    else
    {
        return std::make_unique<ParquetFileWriterWrapper>( schema );
    }
}

SingleFileWriterWrapperContainer::SingleFileWriterWrapperContainer( std::shared_ptr<arrow::Schema> schema, bool isWriteArrowBinary )
        : m_fileWriterWrapper( createSingleFileWrapper( schema, isWriteArrowBinary ) )
{
}

void SingleFileWriterWrapperContainer::SingleFileWriterWrapperContainer::open( const std::string &fileName, const std::string &compression,
                                                                               bool allowOverwrite )
{
    m_fileWriterWrapper -> open( fileName, compression, allowOverwrite );
    setOpen(true);
}

void SingleFileWriterWrapperContainer::close()
{
    m_fileWriterWrapper -> close();
    setOpen(false);
}

void SingleFileWriterWrapperContainer::writeData( const std::vector<std::shared_ptr<ArrowSingleColumnArrayBuilder>> &columnBuilders )
{
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve( columnBuilders.size() );
    for( auto &&columnBuilder:columnBuilders )
    {
        columns.push_back( columnBuilder -> buildArray() );
    }

    auto table = arrow::Table::Make( m_fileWriterWrapper -> getSchema(), columns );

    m_fileWriterWrapper -> writeTable( table );
}


MultipleFileWriterWrapperContainer::MultipleFileWriterWrapperContainer( std::shared_ptr<arrow::Schema> schema, bool isWriteArrowBinary )
{
    auto &fields = schema -> fields();
    m_fileWriterWrappers.reserve( fields.size() );
    for( auto &&field:fields )
    {
        std::vector<std::shared_ptr<arrow::Field>> curFields;
        std::string                                fileExtension = isWriteArrowBinary ? ".arrow" : ".parquet";
        std::string                                fileName      = field -> name() + fileExtension;

        m_fileWriterWrappers.push_back( { fileName,
                                          createSingleFileWrapper( arrow::schema( { field } ), isWriteArrowBinary ) } );
    }
}

void MultipleFileWriterWrapperContainer::open( const std::string &fileName, const std::string &compression, bool allowOverwrite )
{
    for( auto &&record : m_fileWriterWrappers )
    {
        record.m_fileWriterWrapper -> open( fileName + '/' + record.m_fileName,
                                              compression,
                                              allowOverwrite );

    }
    setOpen(true);
}

void MultipleFileWriterWrapperContainer::close()
{
    for( auto &&record : m_fileWriterWrappers )
    {
        record.m_fileWriterWrapper -> close();
    }
    setOpen(false);
}

void MultipleFileWriterWrapperContainer::writeData( const std::vector<std::shared_ptr<ArrowSingleColumnArrayBuilder>> &columnBuilders )
{
    std::vector<std::shared_ptr<arrow::Array>> columns;
    columns.reserve( 1 );
    CSP_TRUE_OR_THROW_RUNTIME( columnBuilders.size() == m_fileWriterWrappers.size(),
                               "Internal error - column builders and file wrappers are expected to have same size" );

    for( unsigned i = 0; i < columnBuilders.size(); ++i )
    {
        columns.clear();
        columns.push_back( columnBuilders[ i ] -> buildArray() );
        auto &fileWriterWrapper = m_fileWriterWrappers[ i ].m_fileWriterWrapper;
        auto table              = arrow::Table::Make( fileWriterWrapper -> getSchema(), columns );
        fileWriterWrapper -> writeTable( table );
    }
}

}