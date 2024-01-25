#include <csp/adapters/parquet/ParquetDictBasketOutputWriter.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <parquet/arrow/writer.h>

namespace csp::adapters::parquet
{
ParquetDictBasketOutputWriter::ParquetDictBasketOutputWriter(
        ParquetOutputAdapterManager *outputAdapterManager,
        const std::string &columnName )
        :
        ParquetWriter( outputAdapterManager, false ), m_nextCycleIndex( 0 )
{
    m_symbolOutputAdapter     = getScalarOutputHandler( CspType::STRING(), columnName + "__csp_symbol" );
    m_cycleIndexOutputAdapter = createScalarOutputHandler( CspType::UINT16(), columnName + "__csp_value_count" );
}

void ParquetDictBasketOutputWriter::start()
{
    ParquetWriter::start();
    m_indexFileWriterContainer = std::make_unique<MultipleFileWriterWrapperContainer>(
            arrow::schema( { arrow::field( m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 ) -> getColumnName(),
                                           m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 ) -> getDataType() ) } ),
            m_adapterMgr.isWriteArrowBinary() );
    if( !m_adapterMgr.getFileName().empty() )
    {
        m_indexFileWriterContainer -> open( m_adapterMgr.getFileName(),
                                            m_adapterMgr.getCompression(), m_adapterMgr.isAllowOverwrite() );

    }
}

void ParquetDictBasketOutputWriter::stop()
{
    auto &&indexColumnArrayBuilder = m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 );
    if( indexColumnArrayBuilder -> length() > 0 )
    {
        CSP_TRUE_OR_THROW_RUNTIME(isFileOpen(), "On stop ParquetDictBasketOutputWriter has data to write but no open file");
        m_indexFileWriterContainer -> writeData( { indexColumnArrayBuilder } );
    }
    m_indexFileWriterContainer -> close();
    m_indexFileWriterContainer = nullptr;

    ParquetWriter::stop();
}

void ParquetDictBasketOutputWriter::writeValue( const std::string &valueKey, const TimeSeriesProvider *ts )
{
    m_adapterMgr.scheduleEndCycle();
    m_symbolOutputAdapter -> writeValue<std::string, StringArrayBuilder>( valueKey );
    ParquetWriter::onEndCycle();
    ++m_nextCycleIndex;
}


void ParquetDictBasketOutputWriter::onEndCycle()
{
    if(isFileOpen())
    {
        m_cycleIndexOutputAdapter -> writeValue<std::uint16_t, UInt16ArrayBuilder>( m_nextCycleIndex );
        auto &&indexColumnArrayBuilder = m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 );
        indexColumnArrayBuilder -> handleRowFinished();
        if( indexColumnArrayBuilder -> length() >= getChunkSize() )
        {
            m_indexFileWriterContainer -> writeData( { indexColumnArrayBuilder } );
        }
        m_nextCycleIndex = 0;
    }
    else
    {
        CSP_TRUE_OR_THROW_RUNTIME(m_nextCycleIndex==0, "ParquetDictBasketOutputWriter has non 0 index with no open file");
    }
}

void ParquetDictBasketOutputWriter::onFileNameChange( const std::string &fileName )
{
    ParquetWriter::onFileNameChange( fileName );
    if( m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 ) -> length() > 0 )
    {
        CSP_TRUE_OR_THROW_RUNTIME( m_indexFileWriterContainer -> isOpen(), "Trying to write basket index data to closed file" );
        m_indexFileWriterContainer -> writeData( { m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 ) } );
    }
    if (m_indexFileWriterContainer->isOpen())
    {
        m_indexFileWriterContainer->close();
    }
    if(!fileName.empty())
    {
        m_indexFileWriterContainer
                -> open( fileName, m_adapterMgr.getCompression(), m_adapterMgr.isAllowOverwrite() );
    }

}

SingleColumnParquetOutputHandler *ParquetDictBasketOutputWriter::createScalarOutputHandler( CspTypePtr type, const std::string &name )
{
    m_allHandlers.push_back( std::make_unique<SingleColumnParquetOutputHandler>( m_engine, *this, type, name ) );
    return static_cast<SingleColumnParquetOutputHandler *>(m_allHandlers.back().get());
}

StructParquetOutputHandler *ParquetDictBasketOutputWriter::createStructOutputHandler( CspTypePtr type,
                                                                                      const DictionaryPtr &fieldMap )
{
    m_allHandlers.push_back( std::make_unique<StructParquetOutputHandler>( m_engine, *this, type, fieldMap ) );
    return static_cast<StructParquetOutputHandler *>(m_allHandlers.back().get());
}


ParquetScalarDictBasketOutputWriter::ParquetScalarDictBasketOutputWriter( ParquetOutputAdapterManager *outputAdapterManager,
                                                                          const std::string &columnName,
                                                                          CspTypePtr cspTypePtr )
        : ParquetDictBasketOutputWriter( outputAdapterManager, columnName )
{
    m_valueOutputAdapter = getScalarOutputHandler( cspTypePtr, columnName );
}

void ParquetScalarDictBasketOutputWriter::writeValue( const std::string &valueKey, const TimeSeriesProvider *ts )
{
    m_valueOutputAdapter -> writeValueFromTs( ts );
    ParquetDictBasketOutputWriter::writeValue(valueKey, ts);
}

ParquetStructDictBasketOutputWriter::ParquetStructDictBasketOutputWriter( ParquetOutputAdapterManager *outputAdapterManager,
                                                                          const std::string &columnName,
                                                                          CspTypePtr cspTypePtr )
        : ParquetDictBasketOutputWriter( outputAdapterManager, columnName )
{

    // We don't support fieldMap for now, only default field map
    auto structMetaPtr = std::static_pointer_cast<const CspStructType>( cspTypePtr ) -> meta().get();
    DictionaryPtr dict = std::make_shared<Dictionary>();
    for(auto&& field:structMetaPtr->fields())
    {
        dict->insert(field->fieldname(), columnName + "." + field->fieldname());
    }
    m_valueOutputAdapter = getStructOutputHandler( cspTypePtr, dict );
}

void ParquetStructDictBasketOutputWriter::writeValue( const std::string &valueKey, const TimeSeriesProvider *ts )
{
    m_valueOutputAdapter -> writeValueFromTs( ts );
    ParquetDictBasketOutputWriter::writeValue(valueKey, ts);
}

}
