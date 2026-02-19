// Per-column readers that extract values from Arrow arrays into CSP struct fields.
//
// FieldReader is the base class with non-virtual bindColumn()/readNext() for
// sequential row processing.  Every reader targets exactly one struct field.
// Scalar readers use the single-column constructor; multi-column readers
// (e.g. NDArray with data + dims) use the multi-column constructor and
// override the virtual bindBatch().

#ifndef _IN_CSP_ADAPTERS_ARROW_ArrowFieldReader_H
#define _IN_CSP_ADAPTERS_ARROW_ArrowFieldReader_H

#include <csp/engine/Struct.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <memory>
#include <string>
#include <vector>

namespace csp::adapters::arrow
{

class FieldReader
{
public:
    virtual ~FieldReader() = default;

    // Constructor for single-column readers
    FieldReader( const std::string & columnName, const StructFieldPtr & field )
        : m_field( field ), m_columnNames( { columnName } )
    {
    }

    // Constructor for multi-column readers (e.g. NDArray with data + dims columns)
    FieldReader( std::vector<std::string> columnNames, const StructFieldPtr & field )
        : m_field( field ), m_columnNames( std::move( columnNames ) )
    {
    }

    // Set primary column pointer and reset row counter. Non-virtual.
    void bindColumn( const ::arrow::Array * column )
    {
        m_column = column;
        m_row = 0;
    }

    // Batch-level bind for readers that need the full RecordBatch (e.g. multi-column
    // numpy readers).  Default does nothing; custom readers override this.
    virtual void bindBatch( const ::arrow::RecordBatch & batch ) {}

    // Read the current row into the struct and advance to the next row.
    void readNext( Struct * s )
    {
        doReadNext( m_row, s );
        ++m_row;
    }

    // Advance the row counter without reading (used to keep child readers in sync
    // when a parent nested struct row is null).
    void skipNext()
    {
        ++m_row;
    }

    // Column names consumed by this reader.
    const std::vector<std::string> & columnNames() const { return m_columnNames; }

protected:
    virtual void doReadNext( int64_t row, Struct * s ) = 0;

    StructFieldPtr             m_field;
    std::vector<std::string>   m_columnNames;
    const ::arrow::Array *     m_column = nullptr;
    int64_t                    m_row    = 0;
};

// Factory: create a scalar FieldReader for a given Arrow field + CSP struct field.
std::unique_ptr<FieldReader> createFieldReader(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const StructFieldPtr & structField
);

}

#endif
