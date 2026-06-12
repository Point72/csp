#include <csp/adapters/parquet/ParquetDictBasketOutputWriter.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <arrow/record_batch.h>

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
    m_indexSchema = ::arrow::schema( { ::arrow::field(
        m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 ) -> getColumnName(),
        m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 ) -> getDataType() ) } );
    if( m_indexSink.onStart )
        m_indexSink.onStart( m_indexSchema );
    auto & fileName = m_adapterMgr.getFileName();
    if( !fileName.empty() && m_indexSink.onFileChange )
        m_indexSink.onFileChange( fileName );
}

void ParquetDictBasketOutputWriter::stop()
{
    auto && indexBuilder = m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 );
    if( indexBuilder -> length() > 0 && m_indexSink.onBatch )
    {
        auto arr = indexBuilder -> buildArray();
        auto rb  = ::arrow::RecordBatch::Make( m_indexSchema, arr -> length(), { arr } );
        m_indexSink.onBatch( rb );
    }
    if( m_indexSink.onStop )
        m_indexSink.onStop();

    ParquetWriter::stop();
}

void ParquetDictBasketOutputWriter::writeValue( const std::string &valueKey, const TimeSeriesProvider *ts )
{
    m_adapterMgr.scheduleEndCycle();
    m_symbolOutputAdapter -> writeValue<std::string>( valueKey );
    ParquetWriter::onEndCycle();
    ++m_nextCycleIndex;
}


void ParquetDictBasketOutputWriter::onEndCycle()
{
    if(isFileOpen())
    {
        m_cycleIndexOutputAdapter -> writeValue<std::uint16_t>( m_nextCycleIndex );
        auto && indexBuilder = m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 );
        indexBuilder -> handleRowFinished();
        if( indexBuilder -> length() >= getChunkSize() && m_indexSink.onBatch )
        {
            auto arr = indexBuilder -> buildArray();
            auto rb  = ::arrow::RecordBatch::Make( m_indexSchema, arr -> length(), { arr } );
            m_indexSink.onBatch( rb );
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
    // Flush any pending index data
    auto && indexBuilder = m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 );
    if( indexBuilder -> length() > 0 && m_indexSink.onBatch )
    {
        auto arr = indexBuilder -> buildArray();
        auto rb  = ::arrow::RecordBatch::Make( m_indexSchema, arr -> length(), { arr } );
        m_indexSink.onBatch( rb );
    }
    if( m_indexSink.onFileChange )
        m_indexSink.onFileChange( fileName );
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
