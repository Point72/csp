#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetOutputAdapterManager_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetOutputAdapterManager_H

#include <csp/adapters/parquet/RecordBatchSink.h>
#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/core/Generator.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <string>
#include <unordered_map>


namespace csp::adapters::parquet
{
class ParquetWriter;

class ParquetOutputFilenameAdapter;

class ParquetDictBasketOutputWriter;

//Top level AdapterManager object for all parquet adapters in the engine
class ParquetOutputAdapterManager final : public csp::AdapterManager
{
public:
    using SinkFactory = std::function<RecordBatchSink( const std::string & )>;

    ParquetOutputAdapterManager( csp::Engine *engine, const Dictionary &properties );
    ~ParquetOutputAdapterManager();

    const char *name() const override{ return "ParquetOutputAdapterManager"; }

    const std::string &getFileName() const{ return m_fileName; }

    const std::string &getTimestampColumnName() const{ return m_timestampColumnName; }

    uint32_t getBatchSize() const{ return m_batchSize; }

    //start the writer, open file if necessary
    void start( DateTime starttime, DateTime endtime ) override;

    //stop the writer, write any unwritten data and close file
    void stop() override;

    DateTime processNextSimTimeSlice( DateTime time ) override;

    OutputAdapter *getOutputAdapter( CspTypePtr &type, const Dictionary &properties );
    OutputAdapter *getListOutputAdapter( CspTypePtr &elemType, const Dictionary &properties );
    ParquetDictBasketOutputWriter *createDictOutputBasketWriter( const char *columnName, const CspTypePtr &cspTypePtr);
    OutputAdapter *createOutputFileNameAdapter();

    void changeFileName( const std::string &filename );

    void scheduleEndCycle();

    void setSink( RecordBatchSink sink );
    void setSinkFactory( SinkFactory factory ) { m_sinkFactory = std::move( factory ); }

private:
    OutputAdapter *getScalarOutputAdapter( CspTypePtr &type, const Dictionary &properties );
    OutputAdapter *getStructOutputAdapter( CspTypePtr &type, const Dictionary &properties );

    std::string                                                 m_fileName;
    std::string                                                 m_timestampColumnName;
    uint32_t                                                    m_batchSize;
    std::unique_ptr<ParquetWriter>                              m_parquetWriter;
    std::unordered_map<std::string, int>                        m_dictBasketWriterIndexByName;
    std::vector<std::unique_ptr<ParquetDictBasketOutputWriter>> m_dictBasketWriters;
    ParquetOutputFilenameAdapter                                *m_outputFilenameAdapter;
    SinkFactory                                                 m_sinkFactory;
};

}

#endif
