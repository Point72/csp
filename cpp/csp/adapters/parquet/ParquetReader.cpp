#include <csp/adapters/parquet/ParquetReader.h>
#include <csp/adapters/parquet/ArrowIPCFileReaderWrapper.h>
#include <csp/adapters/parquet/ParquetReaderColumnAdapter.h>
#include <csp/adapters/parquet/ParquetFileReaderWrapper.h>
#include <csp/adapters/parquet/ParquetStatusUtils.h>
#include <csp/core/FileUtils.h>
#include <csp/core/Exception.h>
#include <csp/core/System.h>
#include <csp/engine/TypeCast.h>
#include <arrow/io/file.h>
#include <arrow/table.h>
#include <arrow/array.h>

namespace
{
struct FieldInfo
{
    std::size_t m_startColumnIndex;
    std::size_t m_width;
};

std::size_t getFieldWidth( std::shared_ptr<::arrow::DataType> fieldType )
{
    if( fieldType -> id() != ::arrow::Type::STRUCT )
    {
        return 1;
    }
    std::size_t res = 0;

    for( auto &childField: fieldType -> fields() )
    {
        res += getFieldWidth( childField -> type() );
    }
    return res;
}

inline std::vector<FieldInfo> getFieldsInfo( const std::vector<std::shared_ptr<::arrow::Field>> &fields )
{
    std::size_t            curStartIndex = 0;
    std::vector<FieldInfo> res;
    res.reserve( fields.size() );
    for( auto &field:fields )
    {
        auto width = getFieldWidth( field -> type() );
        res.push_back( { curStartIndex, width } );
        curStartIndex += width;
    }
    return res;
}

}

namespace csp::adapters::parquet
{

ParquetStructAdapter &ParquetReader::getStructAdapter( const StructAdapterInfo &structAdapterInfo )
{
    auto it = m_structInfoToAdapterIndex.find( structAdapterInfo );
    if( it == m_structInfoToAdapterIndex.end() )
    {
        m_structInfoToAdapterIndex[ structAdapterInfo ] = m_structAdapters.size();
        m_structAdapters.push_back( std::make_unique<ParquetStructAdapter>( *this, structAdapterInfo ) );
        return *m_structAdapters[ m_structAdapters.size() - 1 ];
    }
    else
    {
        return *m_structAdapters[ it -> second ];
    }
}

const utils::Symbol * ParquetReader::getCurSymbol()
{
    if( getSymbolColumnAdapter().valid() )
    {
        switch( m_symbolType )
        {
            case CspType::Type::STRING:
            {
                auto &curSymbol = getSymbolColumnAdapter() -> getCurValue<std::string>();
                CSP_TRUE_OR_THROW_RUNTIME( curSymbol.has_value(),
                                           "Parquet file row contains row with no value for symbol column "
                                           << getSymbolColumnName().value() );
                m_curSymbol = curSymbol.value();
                break;
            }

            case CspType::Type::INT64:
            {
                auto &curSymbol = getSymbolColumnAdapter() -> getCurValue<int64_t>();
                CSP_TRUE_OR_THROW_RUNTIME( curSymbol.has_value(),
                                           "Parquet file row contains row with no value for symbol column "
                                           << getSymbolColumnName().value() );
                m_curSymbol = curSymbol.value();
                break;
            }

            default:
                CSP_THROW( RuntimeException, "Unexpected symbol type: " << m_symbolType );
        }

        return &m_curSymbol;
    }

    return nullptr;
}


void ParquetReader::setSymbolColumnAdapter( ColumnAdapterReference adapter )
{
    m_symbolColumn = adapter;
    try
    {
        m_symbolColumn -> ensureType( CspType::STRING() );
        m_symbolType = CspType::Type::STRING;
    }
    catch( const TypeError & )
    {
        try
        {
            m_symbolColumn -> ensureType( CspType::INT64() );
            m_symbolType = CspType::Type::INT64;
        }
        catch( const TypeError & )
        {
            CSP_THROW( TypeError, "Invalid symbol column type.  Only string and int64 symbols are currently supported" );
        }
    }
}

void ParquetReader::validateSymbolType( const utils::Symbol & symbol )
{
    switch( m_symbolType )
    {
        case CspType::Type::STRING:
            CSP_TRUE_OR_THROW( std::holds_alternative<std::string>( symbol ), TypeError, "Provided symbol type does not match symbol column type (string)" );
            break;

        case CspType::Type::INT64:
            CSP_TRUE_OR_THROW( std::holds_alternative<int64_t>( symbol ), TypeError, "Provided symbol type does not match symbol column type (int64)" );
            break;
        default:
            CSP_THROW( RuntimeException, "Unexpected symbol type: " << m_symbolType );
    }
}

SingleTableParquetReader::SingleTableParquetReader( std::vector<std::string> columns, bool arrowIPC, bool allowMissingColumns,
                                                    std::optional<std::string> symbolColumnName )
        : ParquetReader( symbolColumnName, arrowIPC, allowMissingColumns ), m_columns( columns )
{
}

void SingleTableParquetReader::init()
{
    if(!openNextFile())
    {
        return;
    }

    setColumnAdaptersFromCurrentTable();
}

void SingleTableParquetReader::setColumnAdaptersFromCurrentTable()
{
    m_neededColumnIndices.clear();
    m_columnAdapters.clear();

    m_neededColumnIndices.reserve( m_columns.size() );
    m_columnAdapters.reserve( m_columns.size() );
    auto &fields    = m_schema -> fields();
    auto fieldsInfo = getFieldsInfo( fields );

    for( const auto &columnName:m_columns )
    {
        auto index = m_schema -> GetFieldIndex( columnName );
        auto existingRecordIt = m_columnNameToAdapterIndex.find(columnName);
        if(existingRecordIt != m_columnNameToAdapterIndex.end())
        {
            CSP_TRUE_OR_THROW_RUNTIME( existingRecordIt -> second == m_columnAdapters.size(),
                                       "Unexpected index change of column " << columnName <<
                                                                            " was " << existingRecordIt -> second << " became "
                                                                            << m_columnAdapters.size() );
        }
        else
        {
            m_columnNameToAdapterIndex[ columnName ] = m_columnAdapters.size();
        }
        std::unique_ptr<ParquetColumnAdapter> columnAdapter;

        if( index >= 0 )
        {
            auto &field = fields[ index ];
            columnAdapter = createColumnAdapter( *this, *field, getCurFileOrTableName(), &getStructColumnMeta() );
            auto &fieldInfo = fieldsInfo[ index ];

            for( std::size_t i = 0; i < fieldInfo.m_width; ++i )
            {
                m_neededColumnIndices.push_back( fieldInfo.m_startColumnIndex + i );
            }
        }
        else
        {
            CSP_TRUE_OR_THROW_RUNTIME( isAllowMissingColumns(), "Missing column " << columnName << " in file " << getCurFileOrTableName() );
            columnAdapter = createMissingColumnAdapter( *this, columnName );
        }
        m_columnAdapters.push_back( std::move( columnAdapter ) );
    }

    if( getSymbolColumnName().has_value() )
    {
        setSymbolColumnAdapter( ( *this )[ getSymbolColumnName().value() ] );
    }
}


bool SingleTableParquetReader::start()
{
    return readNextRowGroup() && readNextRow();
}

bool SingleTableParquetReader::skipRow()
{
    return readNextRow();
}

bool SingleTableParquetReader::readNextRow()
{
    if( unlikely( m_curTable == nullptr ) )
    {
        return false;
    }

    if( unlikely( m_curTableNextRow >= m_curTable -> num_rows() ) )
    {
        if( !readNextRowGroup() )
        {
            return false;
        }
    }
    for( auto &colAdapter:m_columnAdapters )
    {
        colAdapter -> readCurValue();
    }
    ++m_curTableNextRow;
    return true;
}


void SingleTableParquetReader::dispatchRow( bool doReadNextRow )
{

    dispatchRow( doReadNextRow, getCurSymbol() );
}

void SingleTableParquetReader::dispatchRow( bool doReadNextRow, const utils::Symbol *symbol )
{
    for( auto &adapter:m_columnAdapters )
    {
        adapter -> dispatchValue( symbol );
    }

    for( auto &adapter:getStructAdapters() )
    {
        adapter -> dispatchValue( symbol );
    }
    if( doReadNextRow )
    {
        readNextRow();
    }
}

bool SingleTableParquetReader::hasData() const
{
    return m_curTable != nullptr;
}

void SingleTableParquetReader::clear()
{
    m_schema = nullptr;
    m_requiredColumnIndices.clear();
    m_curTable = nullptr;
    m_neededColumnIndices.clear();
    m_curTableNextRow = -1;
}

SingleFileParquetReader::SingleFileParquetReader( GeneratorPtr generatorPtr, std::vector<std::string> columns, bool arrowIPC,
                                                  bool allowMissingColumns, bool allowMissingFiles, std::optional<std::string> symbolColumnName )
        : SingleTableParquetReader( columns, arrowIPC, allowMissingColumns, symbolColumnName ), m_generatorPtr( generatorPtr ), m_allowMissingFiles(allowMissingFiles)
{
    init();
}

bool SingleFileParquetReader::openNextFile()
{
    std::string                      fileName;
    FileReaderWrapperPtr             fileReader;
    std::shared_ptr<::arrow::Schema> fileSchema;
    while(true)
    {
        if( m_generatorPtr -> next( fileName ) )
        {
            if(m_allowMissingFiles && !csp::utils::fileExists(fileName))
            {
                continue;
            }
            if( isArrowIPC() )
            {
                fileReader = std::make_unique<ArrowIPCFileReaderWrapper>();
            }
            else
            {
                fileReader = std::make_unique<ParquetFileReaderWrapper>();
            }
            fileReader -> open( fileName );
            fileReader -> getSchema( fileSchema );
            break;
        }
        else
        {
            clear();
            return false;
        }
    }

    auto is_new_schema =  m_schema && !m_schema -> Equals( *fileSchema );
    m_fileName   = fileName;
    m_fileReader = std::move( fileReader );
    m_schema     = fileSchema;

    if(is_new_schema)
    {
        setColumnAdaptersFromCurrentTable();
        resubscribeAll();
    }

    return true;
}

bool SingleFileParquetReader::readNextRowGroup()
{
    if( !m_fileReader )
    {
        return false;
    }
    bool endOfData = false;
    while( !endOfData )
    {
        // Get to first non empty table
        while( m_fileReader -> readNextRowGroup( m_neededColumnIndices, m_curTable ) && m_curTable -> num_rows() == 0 )
        {
        }
        if( m_curTable != nullptr )
        {
            break;
        }
        endOfData = !openNextFile();
    }
    if( endOfData )
    {
        clear();
        return false;
    }
    m_curTableNextRow = 0;
    auto columns = m_curTable -> columns();

    std::size_t columnIndex = 0;
    for( std::size_t i = 0; i < m_columnAdapters.size(); ++i )
    {
        if(!m_columnAdapters[i]->isMissingColumn())
        {
            m_columnAdapters[ i ] -> handleNewBatch( columns[ columnIndex++ ] );
        }
    }
    return true;
}

void SingleFileParquetReader::clear()
{
    SingleTableParquetReader::clear();
    m_fileReader = nullptr;
    m_fileName.clear();
    m_generatorPtr = nullptr;
}

InMemoryTableParquetReader::InMemoryTableParquetReader( GeneratorPtr generatorPtr, std::vector<std::string> columns,
                                                        bool allowMissingColumns, std::optional<std::string> symbolColumnName )
        : SingleTableParquetReader( columns, true, allowMissingColumns, symbolColumnName ), m_generatorPtr( generatorPtr )
{
    init();
}

bool InMemoryTableParquetReader::openNextFile()
{
    std::shared_ptr<::arrow::Schema> schema;
    std::shared_ptr<::arrow::Table>  table;

    if( !m_generatorPtr -> next( table ) )
    {
        clear();
        return false;
    }
    CSP_TRUE_OR_THROW_RUNTIME( table -> num_columns() > 0, "Provided in memory arrow table with 0 columns" );
    schema = table -> schema();

    int refNumChunks = table -> column( 0 ) -> num_chunks();

    for( int i = 0; i < table -> num_columns(); ++i )
    {
        CSP_TRUE_OR_THROW_RUNTIME( table -> column( i ) -> num_chunks() == refNumChunks,
                                   "Found in memory table with non aligned chunks, number of chunks in one column is "
                                           << refNumChunks << " vs " << table -> column( i ) -> num_chunks() << " in another table" );
    }

    auto is_new_schema =  m_schema && !m_schema -> Equals( *schema );

    m_schema         = schema;
    m_fullTable      = table;
    m_nextChunkIndex = 0;
    m_curTable       = nullptr;

    if(is_new_schema)
    {
        setColumnAdaptersFromCurrentTable();
        resubscribeAll();
    }

    return true;
}

bool InMemoryTableParquetReader::readNextRowGroup()
{
    if( !m_fullTable )
    {
        return false;
    }
    bool endOfData = false;

    while( !endOfData )
    {
        if(m_nextChunkIndex >= m_fullTable->column(0)->num_chunks())
        {
            endOfData = !openNextFile();
            continue;
        }

        std::vector<std::shared_ptr<arrow::ChunkedArray>> curTableChunks;
        auto refChunkLength = m_fullTable->column(0)->chunk(m_nextChunkIndex)->length();
        std::vector<std::shared_ptr<arrow::Field>> neededFields;
        for(auto colIndex : m_neededColumnIndices)
        {
            neededFields.push_back(m_fullTable->field(colIndex));
            auto &&curArrayChunk = m_fullTable -> column( colIndex ) -> chunk( m_nextChunkIndex );
            CSP_TRUE_OR_THROW_RUNTIME( curArrayChunk -> length() == refChunkLength,
                                       "Found in memory table with non aligned chunks, for chunk "
                                               << m_nextChunkIndex << "found arrays of lenght " << refChunkLength << " and "
                                               << curArrayChunk -> length() );
            arrow::Result<std::shared_ptr<arrow::ChunkedArray>> newChunkedArrayResult = arrow::ChunkedArray::Make(
                    std::vector<std::shared_ptr<arrow::Array>>{curArrayChunk});
            STATUS_OK_OR_THROW_RUNTIME(newChunkedArrayResult.status(), "Failed to creae a new chunked array");
            curTableChunks.push_back(newChunkedArrayResult.ValueUnsafe());
        }
        m_nextChunkIndex += 1;

        if(refChunkLength > 0)
        {
            m_curTable = arrow::Table::Make(arrow::schema(neededFields), curTableChunks);
            break;
        }
    }
    if( endOfData )
    {
        clear();
        return false;
    }
    m_curTableNextRow = 0;
    auto columns = m_curTable -> columns();

    std::size_t columnIndex = 0;
    for( std::size_t i = 0; i < m_columnAdapters.size(); ++i )
    {
        if(!m_columnAdapters[i]->isMissingColumn())
        {
            m_columnAdapters[ i ] -> handleNewBatch( columns[ columnIndex++ ] );
        }
    }
    return true;
}

void InMemoryTableParquetReader::clear()
{
    try
    {
        SingleTableParquetReader::clear();
    }
    catch( ... )
    {
        m_generatorPtr   = nullptr;
        m_nextChunkIndex = 0;
        m_fullTable      = nullptr;
        throw;
    }
}

MultipleFileParquetReader::MultipleFileParquetReader( FileNameGeneratorReplicator::Ptr generatorReplicatorPtr,
                                                      std::vector<std::string> columns,
                                                      bool arrowIPC,
                                                      bool allowMissingColumns,
                                                      std::optional<std::string> symbolColumnName )
        : ParquetReader( symbolColumnName, arrowIPC, allowMissingColumns ), m_generatorReplicatorPtr( generatorReplicatorPtr )
{
    for( auto &&column : columns )
    {
        m_columnReaders.push_back(
                std::make_unique<SingleFileParquetReader>(
                        m_generatorReplicatorPtr -> getGeneratorReplica( std::string( "/" ) + '/' + column + ".parquet" ),
                        std::vector<std::string>{ column },
                        arrowIPC,
                        allowMissingColumns) );
        m_columnReaderByName[column] = m_columnReaders.back().get();
    }
    if( symbolColumnName.has_value() )
    {
        setSymbolColumnAdapter( ( *this )[ symbolColumnName.value() ] );
    }
}

bool MultipleFileParquetReader::start()
{
    uint32_t successCount = 0;

    for( auto &&columnReader : m_columnReaders )
    {
        if( columnReader -> start() )
        {
            ++successCount;
        }
    }
    CSP_TRUE_OR_THROW_RUNTIME( successCount == 0 || successCount == m_columnReaders.size(),
                               "Expected all or none of the column readers to start, actual:" << successCount << '/'
                                                                                              << m_columnReaders.size() );
    return successCount != 0;
}


ColumnAdapterReference MultipleFileParquetReader::operator[]( const std::string &name )
{
    auto it = m_columnReaderByName.find(name);
    CSP_TRUE_OR_THROW_RUNTIME( it != m_columnReaderByName.end(),
                               "No column " << name << " found in parquet file" );
    return (*it -> second)[name];
}

ParquetColumnAdapter* MultipleFileParquetReader::getCurrentColumnAdapterByIndex( std::size_t index )
{
    CSP_NOT_IMPLEMENTED;
}


bool MultipleFileParquetReader::skipRow()
{
    unsigned successfulSkips = 0;
    for( auto &&columnReader : m_columnReaders )
    {
        if(columnReader -> skipRow())
        {
            ++successfulSkips;
        }
    }
    if(unlikely(successfulSkips == 0))
    {
        return false;
    }
    if(unlikely(successfulSkips!=m_columnReaders.size()))
    {
        CSP_THROW(RuntimeException, "Input files are not alligned - some columns have more data than the others");
    }
    return true;
}

void MultipleFileParquetReader::dispatchRow( bool doReadNextRow )
{
    // By default dispatchRow of columnReaders will dispatch row and read the next row (clear out the currently read row).
    // If we have structAdapters, we can't move on to the next row before we populate the struct adapter fields. So if any
    // struct adapters exist, we need to postpone reading of next row for child readers and explicitly call them after dispatching the
    // struct values.
    auto *symbol = getCurSymbol();
    bool childDoReadNextRow = getStructAdapters().empty() && doReadNextRow && symbol == nullptr;

    for( auto &&columnReader : m_columnReaders )
    {
        columnReader -> dispatchRow( childDoReadNextRow, symbol );
    }
    for( auto &adapter:getStructAdapters() )
    {
        adapter -> dispatchValue( symbol );
    }
    if( !childDoReadNextRow && doReadNextRow )
    {
        for( auto &&columnReader : m_columnReaders )
        {
            columnReader -> readNextRow();
        }
    }
}

bool MultipleFileParquetReader::hasData() const
{
    return !m_columnReaders.empty() && m_columnReaders[ 0 ] -> hasData();
}

void MultipleFileParquetReader::clear()
{
    m_columnReaders.clear();
}


}
