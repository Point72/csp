#ifndef _IN_CSP_ADAPTERS_PARQUET_ParquetReader_H
#define _IN_CSP_ADAPTERS_PARQUET_ParquetReader_H

#include <csp/adapters/utils/StructAdapterInfo.h>
#include <csp/adapters/parquet/FileReaderWrapper.h>
#include <csp/adapters/parquet/ParquetReaderColumnAdapter.h>
#include <csp/adapters/utils/ValueDispatcher.h>
#include <csp/core/Generator.h>
#include <csp/engine/CspType.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <csp/engine/Struct.h>
#include <arrow/type.h>
#include <arrow/table.h>
#include <memory>
#include <parquet/arrow/reader.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

namespace arrow::io
{
class ReadableFile;
}

namespace csp
{
class ManagedSimInputAdapter;
}

namespace csp::adapters::parquet
{

class ParquetColumnAdapter;

class ParquetStructAdapter;

/**
 * Wrapper of "GeneratorPtr". Generator is a general a single generator that provides a sequence of filenames (or folder names) from which the
 * files should be read. Since we might have multiple file readers that need the same sequence of folders, this class wraps the generator
 * and is able to create new generators that will produce exactly the same sequence as the original one. Another "service" that this class
 * provides is that it can append suffix to a given generator replica. For example consider input folders sequence of :
 *  - "folder1"
 *  - "folder2".
 * We can ask for a suffix of "/column1.parquet" when calling to getGeneratorReplica and the replica will produce:
 *  - "folder1/column1.parquet"
 *  - "folder2/column1.parquet".
 */
class FileNameGeneratorReplicator
{
public:
    using Ptr = std::shared_ptr<FileNameGeneratorReplicator>;
    using GeneratorPtr = csp::Generator<std::string, csp::DateTime, csp::DateTime>::Ptr;

    FileNameGeneratorReplicator( GeneratorPtr source )
            : m_generatorPtr( source )
    {
    }

    void init( csp::DateTime start, csp::DateTime end )
    {
        m_generatorPtr -> init( start, end );
    }

    const std::vector<std::string> &getFileNames(){ return m_fileNames; }

    void consumeNextGeneratorFile()
    {
        std::string nextFile;
        if( m_generatorPtr -> next( nextFile ) )
        {
            m_fileNames.push_back( std::move( nextFile ) );
        }
    }

    GeneratorPtr getGeneratorReplica( const std::string &suffix = "" )
    {
        return std::make_shared<ChildGenerator>( *this, suffix );
    }

private:
    class ChildGenerator : public csp::Generator<std::string, csp::DateTime, csp::DateTime>
    {
    public:
        ChildGenerator( FileNameGeneratorReplicator &owner, const std::string &suffix )
                : m_owner( owner ), m_suffix( suffix ), m_nextIndex( 0 )
        {
        }

        void init( csp::DateTime, csp::DateTime ){}

        bool next( std::string &value )
        {
            if( m_nextIndex < 0 )
            {
                return false;
            }
            const std::vector<std::string> &folders = m_owner.getFileNames();
            if( m_nextIndex >= static_cast<int>(folders.size()) )
            {
                m_owner.consumeNextGeneratorFile();
            }
            if( m_nextIndex >= static_cast<int>(folders.size() ) )
            {
                m_nextIndex = -1;
                return false;
            }

            value = folders[ m_nextIndex++ ] + m_suffix;
            return true;
        }

    private:
        FileNameGeneratorReplicator &m_owner;
        const std::string           m_suffix;
        int                         m_nextIndex = 0;
    };

private:
    GeneratorPtr             m_generatorPtr;
    std::vector<std::string> m_fileNames;
};



class ParquetReader;


// When we read multiple files with different schemas, column adapters may change, the adapter reference will persist and still be valid.
class ColumnAdapterReference
{
public:
    ColumnAdapterReference( ParquetReader *reader = nullptr, std::size_t index = -1 ) : m_reader( reader ), m_index( index ){}

    ParquetColumnAdapter &operator*() const;

    ParquetColumnAdapter *get() const{ return &( **this ); };

    ParquetColumnAdapter *operator->() const{ return get(); };

    bool valid() const{ return m_reader != nullptr; }

private:
    ParquetReader *m_reader;
    std::size_t   m_index;
};

class ParquetReader
{
public:
    using StructMetaByColumnName = std::map<std::string, std::shared_ptr<StructMeta>>;

    ParquetReader( std::optional<std::string> symbolColumnName, bool arrowIPC, bool allowMissingColumns )
            : m_symbolColumnName( symbolColumnName ), m_arrowIPC( arrowIPC ), m_allowMissingColumns(allowMissingColumns)
    {
    }

    virtual ~ParquetReader() = default;
    ParquetReader( const ParquetReader &other ) = delete;
    ParquetReader &operator=( const ParquetReader &other ) = delete;

    inline bool isAllowMissingColumns() const
    {
        return m_allowMissingColumns;
    }

    virtual bool start() = 0;

    virtual bool skipRow() = 0;

    bool skipRows( int nRows )
    {
        for( int i = 0; i < nRows; ++i )
        {
            if(unlikely(!skipRow()))
            {
                return false;
            }
        }
        return true;
    }

    void dispatchRow()
    {
        dispatchRow( true );
    }

    virtual void dispatchRow( bool readNextRow ) = 0;

    // Check if the reader can't read any data at all (no input files). In that case even the file schema
    // can't be deduces
    virtual bool isEmpty() const { return false; }
    // Check if the reader has any more data left to read. Note it's different from "isEmpty", isEmpty, will always return
    // the same value, while this function will return false only when all data that could be read is already read. This
    // function should be called only after start, before that it is not expected to be valid.
    virtual bool hasData() const = 0;

    ParquetStructAdapter &getStructAdapter( const csp::adapters::utils::StructAdapterInfo &structAdapterInfo );

    virtual ColumnAdapterReference operator[]( const std::string &name ) = 0;

    virtual ParquetColumnAdapter* getCurrentColumnAdapterByIndex( std::size_t index ) = 0;

    virtual int64_t getCurRow() const = 0;

    bool isArrowIPC() const{ return m_arrowIPC; }

    virtual std::string getCurFileOrTableName() const = 0;

    virtual void addSubscriber( const std::string &column, ManagedSimInputAdapter *inputAdapter,
                                const std::optional<utils::Symbol> &symbol )
    {
        if( symbol )
            validateSymbolType( symbol.value() );
        (*this)[column]->addSubscriber( inputAdapter, symbol );
    }

    virtual void addListSubscriber( const std::string &column, ManagedSimInputAdapter *inputAdapter,
                                    const std::optional<utils::Symbol> &symbol, const DialectGenericListReaderInterface::Ptr &listReaderInterface )
    {
        if( symbol )
            validateSymbolType( symbol.value() );
        (*this)[column]->addSubscriber( inputAdapter, symbol, listReaderInterface );
    }

    /**
     * Add parquet struct adapter that depends on columns from "this" and needs to be updated if the schema of the read files changes.
     * @param adapter
     */
    void addDependentStructAdapter(ParquetStructAdapter* adapter) {m_dependentAdapters.insert(adapter);}

    /**
     * Set a known schema of a struct column. Any future files that will have the column will automatically bind to the given csp struct
     * metadata
     * @param columnName
     * @param metaData
     */
    void setStructColumnMeta(const std::string& columnName, const std::shared_ptr<StructMeta>& metaData)
    {
        m_structMetaByColumnName[columnName] = metaData;
    }

    /**
     * @return The known mappings of struct columns to the corresponding csp metadata. NOTE: the values might not be set for all struct columns,
     * currently the values are set only for struct columns that are used as fields inside a wrapping struct (i.e only for columns which are
     * user as substructs).
     */
    const StructMetaByColumnName& getStructColumnMeta() const
    {
        return m_structMetaByColumnName;
    }
protected:
    using StructAdapterInfo = csp::adapters::utils::StructAdapterInfo;

    using StructAdapterContainer = std::vector<std::unique_ptr<ParquetStructAdapter>>;
    using StructInfoToIndexMap = std::unordered_map<StructAdapterInfo, std::size_t>;

    inline const StructAdapterContainer &getStructAdapters() const
    {
        return m_structAdapters;
    }

    const utils::Symbol * getCurSymbol();

    void setSymbolColumnAdapter( ColumnAdapterReference adapter );

    ColumnAdapterReference &getSymbolColumnAdapter(){ return m_symbolColumn; }

    const std::optional<std::string> &getSymbolColumnName() const{ return m_symbolColumnName; }
    const std::set<ParquetStructAdapter *>& getDependentAdapters() const {return m_dependentAdapters;}
private:
    void validateSymbolType( const utils::Symbol & symbol );

    StructAdapterContainer           m_structAdapters;
    StructInfoToIndexMap             m_structInfoToAdapterIndex;
    std::optional<std::string>       m_symbolColumnName;
    ColumnAdapterReference           m_symbolColumn;
    bool                             m_arrowIPC            = false;
    bool                             m_allowMissingColumns = false;
    CspType::Type                    m_symbolType;
    utils::Symbol                    m_curSymbol;
    std::set<ParquetStructAdapter *> m_dependentAdapters;
    StructMetaByColumnName           m_structMetaByColumnName;
};

inline ParquetColumnAdapter & ColumnAdapterReference::operator*() const
{
    return *( m_reader -> getCurrentColumnAdapterByIndex( m_index ) );
}

struct ColumnSubscriberInfo
{
    ManagedSimInputAdapter     * m_inputAdapter;
    std::optional<utils::Symbol> m_symbol;
};

struct ListColumnSubscriberInfo : public ColumnSubscriberInfo
{
    const DialectGenericListReaderInterface::Ptr listReader;
};

struct ColumnSubscriptionContainer
{
    std::map<std::string, std::vector<ColumnSubscriberInfo>> m_scalarColumnSubscriptions;
    std::map<std::string, std::vector<ListColumnSubscriberInfo>> m_listColumnSubscriptions;
};

class SingleTableParquetReader : public ParquetReader
{
public:
    using NewBatchCallback = std::function<void( std::shared_ptr<::arrow::Table> & )>;
    using NewRowCallback = std::function<void( std::size_t )>;

    SingleTableParquetReader( std::vector<std::string> columns, bool arrowIPC, bool allowMissingColumns,
                              std::optional<std::string> symbolColumnName = {} );
    bool start() override;

    bool readNextRow();

    bool skipRow() override;

    using ParquetReader::dispatchRow;
    void dispatchRow( bool doReadNextRow ) override;
    void dispatchRow( bool doReadNextRow, const utils::Symbol *symbol );

    bool isEmpty() const override {return m_columnAdapters.empty();}
    bool hasData() const override;

    ColumnAdapterReference operator[]( const std::string &name ) override
    {
        auto it = m_columnNameToAdapterIndex.find( name );
        CSP_TRUE_OR_THROW_RUNTIME( it != m_columnNameToAdapterIndex.end(),
                                   "No column " << name << " found in parquet file" );
        return ColumnAdapterReference(this, it->second);
    }

    ParquetColumnAdapter* getCurrentColumnAdapterByIndex( std::size_t index ) override
    {
        return m_columnAdapters[index].get();
    }

    int64_t getCurRow() const override{ return m_curTableNextRow; }


    void addSubscriber( const std::string &column, ManagedSimInputAdapter *inputAdapter,
                        const std::optional<utils::Symbol> &symbol ) override

    {
        ParquetReader::addSubscriber( column, inputAdapter, symbol );
        m_columnSubscriptionContainer.m_scalarColumnSubscriptions[column].push_back(ColumnSubscriberInfo{inputAdapter, symbol});
    }

    virtual void addListSubscriber( const std::string &column, ManagedSimInputAdapter *inputAdapter,
                                    const std::optional<utils::Symbol> &symbol, const DialectGenericListReaderInterface::Ptr &listReaderInterface ) override
    {
        ParquetReader::addListSubscriber( column, inputAdapter, symbol, listReaderInterface);
        m_columnSubscriptionContainer.m_listColumnSubscriptions[column].push_back(ListColumnSubscriberInfo{{inputAdapter, symbol}, listReaderInterface});
    }

protected:
    void init();
    void setColumnAdaptersFromCurrentTable();
    virtual bool openNextFile() = 0;
    virtual bool readNextRowGroup() = 0;
    virtual void clear();
    void resubscribeAll()
    {
        for(auto&& column_vector_pair:m_columnSubscriptionContainer.m_scalarColumnSubscriptions)
        {
            for(auto&& record:column_vector_pair.second)
            {
                // We need to call the base class subscribe since we don't want on resubscription to add columns again to the container
                ParquetReader::addSubscriber(column_vector_pair.first, record.m_inputAdapter, record.m_symbol);
            }
        }
        for(auto&& column_vector_pair:m_columnSubscriptionContainer.m_listColumnSubscriptions)
        {
            for(auto&& record:column_vector_pair.second)
            {
                // We need to call the base class subscribe since we don't want on resubscription to add columns again to the container
                ParquetReader::addListSubscriber(column_vector_pair.first, record.m_inputAdapter, record.m_symbol, record.listReader);
            }
        }
        for(auto&& dependentAdapter: getDependentAdapters())
        {
            dependentAdapter->onSchemaChange();
        }
    }


protected:
    std::vector<std::string>                           m_columns;
    std::vector<std::unique_ptr<ParquetColumnAdapter>> m_columnAdapters;
    std::unordered_map<std::string, std::size_t>       m_columnNameToAdapterIndex;
    std::shared_ptr<::arrow::Schema>                   m_schema;
    std::vector<int>                                   m_requiredColumnIndices;
    std::shared_ptr<::arrow::Table>                    m_curTable;
    std::vector<int>                                   m_neededColumnIndices;
    int64_t                                            m_curTableNextRow = -1;
    ColumnSubscriptionContainer                        m_columnSubscriptionContainer;
};

class SingleFileParquetReader final : public SingleTableParquetReader
{
public:
    using GeneratorPtr = csp::Generator<std::string, csp::DateTime, csp::DateTime>::Ptr;

    SingleFileParquetReader( GeneratorPtr generatorPtr, std::vector<std::string> columns, bool arrowIPC, bool allowMissingColumns,
                             bool allowMissingFiles = false, std::optional<std::string> symbolColumnName = {} );

    std::string getCurFileOrTableName() const override{ return m_fileName; }

protected:
    bool openNextFile() override;
    bool readNextRowGroup() override;

    void clear() override;

private:
    GeneratorPtr         m_generatorPtr;
    std::string          m_fileName;
    FileReaderWrapperPtr m_fileReader;
    bool                 m_allowMissingFiles;
};

class InMemoryTableParquetReader final : public SingleTableParquetReader
{
public:
    using GeneratorPtr = csp::Generator<std::shared_ptr<arrow::Table>, csp::DateTime, csp::DateTime>::Ptr;

    InMemoryTableParquetReader( GeneratorPtr generatorPtr, std::vector<std::string> columns,
                                bool allowMissingColumns,
                                std::optional<std::string> symbolColumnName = {} );
    std::string getCurFileOrTableName() const override{ return "IN_MEMORY_TABLE"; }

protected:
    bool openNextFile() override;
    bool readNextRowGroup() override;

    void clear() override;

private:
    GeneratorPtr                    m_generatorPtr;
    std::shared_ptr<::arrow::Table> m_fullTable;
    int64_t                         m_nextChunkIndex = 0;
};

class MultipleFileParquetReader final : public ParquetReader
{
public:
    using NewBatchCallback = std::function<void( std::shared_ptr<::arrow::Table> & )>;
    using NewRowCallback = std::function<void( std::size_t )>;
    using GeneratorPtr = csp::Generator<std::string, csp::DateTime, csp::DateTime>::Ptr;

    MultipleFileParquetReader( FileNameGeneratorReplicator::Ptr generatorReplicatorPtr, std::vector<std::string> columns, bool arrowIPC,
                               bool allowMissingColumns, std::optional<std::string> symbolColumnName = {} );

    bool start() override;

    bool skipRow() override;

    using ParquetReader::dispatchRow;
    void dispatchRow( bool doReadNextRow ) override;

    bool hasData() const override;

    ColumnAdapterReference operator[]( const std::string &name ) override;
    ParquetColumnAdapter* getCurrentColumnAdapterByIndex( std::size_t index ) override;

    int64_t getCurRow() const override{ CSP_NOT_IMPLEMENTED; }

    std::string getCurFileOrTableName() const override{ return "MULTI_FILE_READER"; }
private:
    void clear();

private:
    FileNameGeneratorReplicator::Ptr                           m_generatorReplicatorPtr;
    std::vector<std::unique_ptr<SingleFileParquetReader>>      m_columnReaders;
    std::unordered_map<std::string, SingleFileParquetReader *> m_columnReaderByName;
};

}


#endif //_IN_CSP_ADAPTERS_PARQUET_ParquetIterator_H
