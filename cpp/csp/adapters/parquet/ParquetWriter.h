#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetWriter_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetWriter_H

#include <csp/adapters/parquet/ParquetOutputAdapterManager.h>
#include <csp/adapters/parquet/RecordBatchSink.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Engine.h>
#include <arrow/record_batch.h>
#include <string>
#include <optional>
#include <unordered_map>

namespace arrow{ class KeyValueMetadata; class Schema; }

namespace csp::adapters::parquet
{
class ArrowBackedArrayBuilder;
class ArrowSingleColumnArrayBuilder;

class ParquetOutputHandler;
class SingleColumnParquetOutputHandler;
class SingleColumnParquetOutputAdapter;
class StructParquetOutputHandler;
class StructParquetOutputAdapter;
class ListColumnParquetOutputHandler;
class ListColumnParquetOutputAdapter;


class ParquetWriter : public EndCycleListener
{
public:
    ParquetWriter( ParquetOutputAdapterManager *mgr, std::optional<bool> writeTimestampColumn = {} );
    ParquetWriter( ParquetOutputAdapterManager *mgr, const Dictionary & properties );

    ~ParquetWriter();

    void setSink( RecordBatchSink sink ) { m_sink = std::move( sink ); }

    SingleColumnParquetOutputAdapter *getScalarOutputAdapter( CspTypePtr &type, const std::string &columnName );
    StructParquetOutputAdapter *getStructOutputAdapter( CspTypePtr &type, csp::DictionaryPtr fieldMap );
    ListColumnParquetOutputAdapter *getListOutputAdapter( CspTypePtr &elemType, const std::string &columnName );

    SingleColumnParquetOutputHandler *getScalarOutputHandler( CspTypePtr &type, const std::string &columnName );
    StructParquetOutputHandler *getStructOutputHandler( CspTypePtr &type, csp::DictionaryPtr fieldMap );
    ListColumnParquetOutputHandler *getListOutputHandler( CspTypePtr &elemType, const std::string &columnName );

    PushInputAdapter *getStatusAdapter();

    virtual void start();
    virtual void stop();

    void onEndCycle() override;

    std::uint32_t getChunkSize() const{ return m_adapterMgr.getBatchSize(); }

    virtual void scheduleEndCycleEvent()
    {
        m_adapterMgr.scheduleEndCycle();
    }

    bool isFileOpen() const { return m_fileOpen; }

    virtual void onFileNameChange(const std::string& fileName);
protected:
    virtual SingleColumnParquetOutputHandler *createScalarOutputHandler( CspTypePtr type, const std::string& name );
    virtual ListColumnParquetOutputHandler *createListOutputHandler( CspTypePtr &elemType, const std::string &columnName );
    virtual StructParquetOutputHandler *createStructOutputHandler( CspTypePtr type, const DictionaryPtr &fieldMap );

private:
    void flushBatch();
    std::shared_ptr<::arrow::RecordBatch> buildRecordBatch();
protected:
    using Adapters = std::vector<ParquetOutputHandler *>;
    using PublishedColumnNames = std::unordered_set<std::string>;

    ParquetOutputAdapterManager &m_adapterMgr;
    Engine                      *m_engine;
    ArrowBackedArrayBuilder     *m_timestampBuilder = nullptr;
private:
    Adapters             m_adapters;
    PublishedColumnNames m_publishedColumnNames;

    RecordBatchSink                                             m_sink;
    bool                                                        m_fileOpen = false;

    std::shared_ptr<::arrow::Schema>                            m_schema;
    std::vector<std::shared_ptr<ArrowSingleColumnArrayBuilder>> m_columnBuilders;
    std::uint32_t                                               m_curChunkSize;
    std::optional<bool>                                         m_writeTimestampColumn;

    std::shared_ptr<::arrow::KeyValueMetadata>                                 m_fileMetaData;
    std::unordered_map<std::string,std::shared_ptr<::arrow::KeyValueMetadata>> m_columnMetaData;
};

}

#endif
