#include <csp/adapters/parquet/ParquetDictBasketOutputWriter.h>
#include <csp/adapters/parquet/ParquetOutputAdapter.h>
#include <arrow/record_batch.h>
#include <exception>

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
        m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 ) -> getDataType() ) }, getFileMetaData() );
    if( m_indexSink )
        m_indexSink -> onStart( m_indexSchema );
    auto & fileName = m_adapterMgr.getFileName();
    if( !fileName.empty() && m_indexSink )
        m_indexSink -> onFileChange( fileName );
}

void ParquetDictBasketOutputWriter::stop()
{
    // Exception-safe teardown: a failure writing/closing the index sink must not skip the value+symbol
    // writer's stop() (which flushes the final batch and writes its footer) -- otherwise an index-side
    // I/O error would leave the main basket data file unreadable. Attempt every step, rethrow first.
    std::exception_ptr firstError;
    auto guard = [&firstError]( auto && fn )
    {
        try { fn(); }
        catch( ... ) { if( !firstError ) firstError = std::current_exception(); }
    };

    guard( [&] { flushIndexBatch(); } );
    if( m_indexSink )
        guard( [&] { m_indexSink -> onStop(); } );

    guard( [&] { ParquetWriter::stop(); } );

    if( firstError )
        std::rethrow_exception( firstError );
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
        if( indexBuilder -> length() >= getChunkSize() )
            flushIndexBatch();
        m_nextCycleIndex = 0;
    }
    else
    {
        CSP_TRUE_OR_THROW_RUNTIME(m_nextCycleIndex==0, "ParquetDictBasketOutputWriter has non 0 index with no open file");
    }
}

void ParquetDictBasketOutputWriter::onFileNameChange( const std::string &fileName )
{
    // Rotate the main (value+symbol) writer, then flush pending index data and rotate the index file.
    // Each step is guarded so a failure in one cannot leave the index file un-rotated and
    // desynchronised from the main file; the first error is rethrown after all steps are attempted.
    std::exception_ptr firstError;
    auto guard = [&firstError]( auto && fn )
    {
        try { fn(); }
        catch( ... ) { if( !firstError ) firstError = std::current_exception(); }
    };

    guard( [&] { ParquetWriter::onFileNameChange( fileName ); } );

    // Flush any pending index data, then rotate the index file.
    guard( [&] { flushIndexBatch(); } );
    if( m_indexSink )
        guard( [&] { m_indexSink -> onFileChange( fileName ); } );

    if( firstError )
        std::rethrow_exception( firstError );
}

void ParquetDictBasketOutputWriter::flushIndexBatch()
{
    auto && indexBuilder = m_cycleIndexOutputAdapter -> getColumnArrayBuilder( 0 );
    if( indexBuilder -> length() > 0 && m_indexSink )
    {
        auto arr = indexBuilder -> buildArray();
        auto rb  = ::arrow::RecordBatch::Make( m_indexSchema, arr -> length(), { arr } );
        m_indexSink -> onBatch( rb );
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

    // We don't support fieldMap for now, only default field map.
    // Iterate fieldNames() (declaration order), NOT fields() (sorted by memory layout): csp writes
    // struct columns -- both top-level and nested -- in declaration order, so the on-disk column order
    // matches the struct definition and stays stable regardless of internal field packing.
    auto structMetaPtr = std::static_pointer_cast<const CspStructType>( cspTypePtr ) -> meta().get();
    DictionaryPtr dict = std::make_shared<Dictionary>();
    for( auto && fieldName : structMetaPtr -> fieldNames() )
    {
        dict->insert( fieldName, columnName + "." + fieldName );
    }
    m_valueOutputAdapter = getStructOutputHandler( cspTypePtr, dict );
}

void ParquetStructDictBasketOutputWriter::writeValue( const std::string &valueKey, const TimeSeriesProvider *ts )
{
    m_valueOutputAdapter -> writeValueFromTs( ts );
    ParquetDictBasketOutputWriter::writeValue(valueKey, ts);
}

}
