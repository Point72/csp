// Implementation of RecordBatchToStructConverter.

#include <csp/adapters/arrow/RecordBatchToStruct.h>
#include <csp/engine/CspType.h>

#include <arrow/type.h>

#include <unordered_set>

namespace csp::adapters::arrow
{

namespace
{

// Helper to resolve column name -> struct field name mapping
// If fieldMap is null, column name = field name (identity mapping)
std::string resolveFieldName( const DictionaryPtr & fieldMap, const std::string & columnName )
{
    if( !fieldMap )
        return columnName;

    std::string fieldName;
    if( fieldMap -> tryGet<std::string>( columnName, fieldName ) )
        return fieldName;

    return columnName;
}

} // anonymous namespace

RecordBatchToStructConverter::RecordBatchToStructConverter(
    const std::shared_ptr<::arrow::Schema> & schema,
    const std::shared_ptr<StructMeta> & structMeta,
    const DictionaryPtr & fieldMap,
    std::vector<std::unique_ptr<FieldReader>> customReaders )
    : m_structMeta( structMeta )
{
    // Build a set of column names handled by custom readers so we skip them in the scalar loop
    std::unordered_set<std::string> customColumnNames;
    for( auto & cr : customReaders )
        for( auto & name : cr -> columnNames() )
            customColumnNames.insert( name );

    // Build scalar field readers from schema fields
    for( int i = 0; i < schema -> num_fields(); ++i )
    {
        auto arrowField = schema -> field( i );

        // Skip columns handled by custom readers
        if( customColumnNames.count( arrowField -> name() ) )
            continue;

        // Skip columns that don't have a matching struct field
        std::string fieldName = resolveFieldName( fieldMap, arrowField -> name() );
        auto structField = structMeta -> field( fieldName );
        if( !structField )
            continue;

        m_scalarReaders.push_back( { createFieldReader( arrowField, structField ), i } );
    }

    // Store custom readers separately
    m_customReaders = std::move( customReaders );
}

std::vector<StructPtr> RecordBatchToStructConverter::convert( const ::arrow::RecordBatch & batch )
{
    int64_t numRows = batch.num_rows();

    // Phase 1: pre-allocate all structs
    std::vector<StructPtr> result;
    result.reserve( numRows );
    for( int64_t i = 0; i < numRows; ++i )
        result.push_back( m_structMeta -> create() );

    // Phase 2: columnar read — one readAll() call per column
    for( auto & entry : m_scalarReaders )
    {
        entry.reader -> bindColumn( batch.column( entry.columnIndex ).get() );
        entry.reader -> readAll( result, numRows );
    }

    for( auto & reader : m_customReaders )
    {
        reader -> bindBatch( batch );
        reader -> readAll( result, numRows );
    }

    return result;
}

}
