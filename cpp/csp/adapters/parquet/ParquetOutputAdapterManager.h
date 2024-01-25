#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetOutputAdapterManager_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetOutputAdapterManager_H

#include <csp/adapters/parquet/ParquetReader.h>
#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/core/Generator.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <set>
#include <string>
#include <unordered_map>
#include <csp/adapters/parquet/DialectGenericListWriterInterface.h>


namespace csp::adapters::parquet
{
class ParquetWriter;

class ParquetOutputFilenameAdapter;

class ParquetDictBasketOutputWriter;

//Top level AdapterManager object for all parquet adapters in the engine
class ParquetOutputAdapterManager final : public csp::AdapterManager
{
public:
    using FileVisitorCallback = std::function<void(const std::string &)>;

    ParquetOutputAdapterManager( csp::Engine *engine, const Dictionary &properties, FileVisitorCallback fileVisitor );
    ~ParquetOutputAdapterManager();

    const char *name() const override{ return "ParquetOutputAdapterManager"; }

    const std::string &getFileName() const{ return m_fileName; }

    const std::string &getTimestampColumnName() const{ return m_timestampColumnName; }

    bool isAllowOverwrite() const{ return m_allowOverwrite; }

    uint32_t getBatchSize() const{ return m_batchSize; }

    std::string getCompression() const{ return m_compression; }

    bool isWriteArrowBinary() const{ return m_writeArrowBinary; }

    bool isSplitColumnsToFiles() const{ return m_splitColumnsToFiles; }

    //start the writer, open file if necessary
    void start( DateTime starttime, DateTime endtime ) override;

    //stop the writer, write any unwritten data and close file
    void stop() override;

    DateTime processNextSimTimeSlice( DateTime time ) override;

    OutputAdapter *getOutputAdapter( CspTypePtr &type, const Dictionary &properties );
    OutputAdapter *getListOutputAdapter( CspTypePtr &elemType, const Dictionary &properties,
                                         const DialectGenericListWriterInterface::Ptr& listWriterInterface );
    ParquetDictBasketOutputWriter *createDictOutputBasketWriter( const char *columnName, const CspTypePtr &cspTypePtr);
    OutputAdapter *createOutputFileNameAdapter();

    void changeFileName( const std::string &filename );

    void scheduleEndCycle();

private:
    OutputAdapter *getScalarOutputAdapter( CspTypePtr &type, const Dictionary &properties );
    OutputAdapter *getStructOutputAdapter( CspTypePtr &type, const Dictionary &properties );

    std::string                                                 m_fileName;
    std::string                                                 m_timestampColumnName;
    bool                                                        m_allowOverwrite;
    uint32_t                                                    m_batchSize;
    std::string                                                 m_compression;
    bool                                                        m_writeArrowBinary;
    bool                                                        m_splitColumnsToFiles;
    std::unique_ptr<ParquetWriter>                              m_parquetWriter;
    std::unordered_map<std::string, int>                        m_dictBasketWriterIndexByName;
    std::vector<std::unique_ptr<ParquetDictBasketOutputWriter>> m_dictBasketWriters;
    FileVisitorCallback                                         m_fileVisitor;
    ParquetOutputFilenameAdapter                                *m_outputFilenameAdapter;
};

}

#endif
