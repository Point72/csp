// Implementation of StructToRecordBatchConverter.

#include <csp/adapters/arrow/StructToRecordBatch.h>
#include <csp/engine/CspType.h>

#include <arrow/builder.h>
#include <arrow/type.h>

namespace csp::adapters::arrow
{

StructToRecordBatchConverter::StructToRecordBatchConverter(
    const std::shared_ptr<StructMeta> & structMeta,
    const DictionaryPtr & fieldMap,
    std::vector<std::unique_ptr<FieldWriter>> customWriters )
    : m_structMeta( structMeta )
{
    std::vector<std::shared_ptr<::arrow::Field>> arrowFields;

    if( fieldMap )
    {
        // When fieldMap is provided, only include fields listed in it
        for( auto it = fieldMap -> begin(); it != fieldMap -> end(); ++it )
        {
            auto fieldName = it.key();
            auto colName   = it.value<std::string>();
            auto structField = structMeta -> field( fieldName );
            if( !structField )
                continue;

            // Skip DIALECT_GENERIC fields (handled by custom writers)
            if( structField -> type() -> type() == CspType::Type::DIALECT_GENERIC )
                continue;

            auto created = createFieldWriter( colName, structField );
            for( auto & dt : created.writer -> dataTypes() )
                arrowFields.push_back( std::make_shared<::arrow::Field>( colName, dt ) );
            m_writers.push_back( std::move( created.writer ) );
        }
    }
    else
    {
        // No fieldMap: include all non-DIALECT_GENERIC fields using fieldNames() for stable insertion order
        // (fields() is sorted by type/size for memory layout optimization, not insertion order)
        for( auto & fieldName : structMeta -> fieldNames() )
        {
            auto structField = structMeta -> field( fieldName );
            if( !structField || structField -> type() -> type() == CspType::Type::DIALECT_GENERIC )
                continue;

            auto created = createFieldWriter( fieldName, structField );
            for( auto & dt : created.writer -> dataTypes() )
                arrowFields.push_back( std::make_shared<::arrow::Field>( fieldName, dt ) );
            m_writers.push_back( std::move( created.writer ) );
        }
    }

    // Append custom writers and their columns to schema
    for( auto & cw : customWriters )
    {
        auto & names = cw -> columnNames();
        auto & types = cw -> dataTypes();
        CSP_TRUE_OR_THROW_RUNTIME( names.size() == types.size(),
                                   "FieldWriter columnNames and dataTypes must have the same size" );
        for( size_t i = 0; i < names.size(); ++i )
            arrowFields.push_back( std::make_shared<::arrow::Field>( names[i], types[i] ) );
        m_writers.push_back( std::move( cw ) );
    }

    m_schema = std::make_shared<::arrow::Schema>( arrowFields );
}

std::vector<std::shared_ptr<::arrow::RecordBatch>> StructToRecordBatchConverter::convert(
    const std::vector<StructPtr> & structs, int64_t maxBatchSize )
{
    int64_t totalRows = static_cast<int64_t>( structs.size() );
    if( maxBatchSize <= 0 )
        maxBatchSize = totalRows;

    std::vector<std::shared_ptr<::arrow::RecordBatch>> batches;

    for( int64_t offset = 0; offset < totalRows; offset += maxBatchSize )
    {
        int64_t chunkRows = std::min( maxBatchSize, totalRows - offset );

        for( auto & writer : m_writers )
            writer -> reserve( chunkRows );

        for( int64_t i = offset; i < offset + chunkRows; ++i )
        {
            const Struct * rawPtr = structs[i].get();
            for( auto & writer : m_writers )
                writer -> writeNext( rawPtr );
        }

        std::vector<std::shared_ptr<::arrow::Array>> arrays;
        arrays.reserve( m_schema -> num_fields() );
        for( auto & writer : m_writers )
        {
            auto writerArrays = writer -> finish();
            arrays.insert( arrays.end(),
                           std::make_move_iterator( writerArrays.begin() ),
                           std::make_move_iterator( writerArrays.end() ) );
        }

        batches.push_back( ::arrow::RecordBatch::Make( m_schema, chunkRows, std::move( arrays ) ) );
    }

    return batches;
}

}
