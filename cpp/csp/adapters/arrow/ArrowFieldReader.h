// Per-column readers that extract values from Arrow arrays.
//
// FieldReader is the base class with non-virtual bindColumn()/readNext() for
// sequential row processing.  Readers can write to a Struct field (readNext)
// or to a raw value pointer (readNextValue) for use without structs.
// Scalar readers use the single-column constructor; multi-column readers
// (e.g. NDArray with data + dims) use the multi-column constructor and
// override the virtual bindBatch().

#ifndef _IN_CSP_ADAPTERS_ARROW_ArrowFieldReader_H
#define _IN_CSP_ADAPTERS_ARROW_ArrowFieldReader_H

#include <csp/engine/Struct.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <functional>
#include <memory>
#include <optional>
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

    // Read the current row's value into an optional<T> (passed as void*) and advance.
    // The target must point to a std::optional<T> matching this reader's value type.
    // Null arrow values → optional.reset(); valid → optional = extracted value.
    void readNextValue( void * optionalOut )
    {
        doReadNextValue( m_row, optionalOut );
        ++m_row;
    }

    // Advance the row counter without reading (used to keep child readers in sync
    // when a parent nested struct row is null).
    virtual void skipNext()
    {
        ++m_row;
    }

    // Columnar bulk-read: read all rows for this column into pre-allocated structs.
    // Default implementation loops over doReadNext(); concrete readers override
    // with a null_count==0 fast path to skip per-row validity checks.
    virtual void readAll( std::vector<StructPtr> & structs, int64_t numRows )
    {
        for( int64_t row = 0; row < numRows; ++row )
            doReadNext( row, structs[row].get() );
        m_row = numRows;
    }

    // Column names consumed by this reader.
    const std::vector<std::string> & columnNames() const { return m_columnNames; }

    // The struct field this reader targets (may be null for fieldless readers).
    const StructFieldPtr & field() const { return m_field; }

protected:
    // Core extraction: returns true if the arrow value at row is non-null and
    // writes the extracted value into valueOut (a raw T*).  Subclasses implement
    // doExtract; doReadNext and doReadNextValue are provided by TypedFieldReader.
    virtual void doReadNext( int64_t row, Struct * s ) = 0;
    virtual void doReadNextValue( int64_t row, void * optionalOut ) = 0;

    StructFieldPtr             m_field;
    std::vector<std::string>   m_columnNames;
    const ::arrow::Array *     m_column = nullptr;
    int64_t                    m_row    = 0;
};

// Typed intermediate: provides default doReadNext/doReadNextValue from a single
// doExtract(row, ValueT&) → bool.  Concrete readers inherit this and only
// implement doExtract (+ optionally readAll for the bulk columnar fast path).
template<typename ValueT>
class TypedFieldReader : public FieldReader
{
public:
    using FieldReader::FieldReader;  // inherit constructors

protected:
    // Single extraction point: returns true if value is non-null, writes into out.
    virtual bool doExtract( int64_t row, ValueT & out ) = 0;

    void doReadNext( int64_t row, Struct * s ) override
    {
        ValueT tmp{};
        if( doExtract( row, tmp ) )
            m_field -> template setValue<ValueT>( s, std::move( tmp ) );
    }

    void doReadNextValue( int64_t row, void * optionalOut ) override
    {
        auto & out = *static_cast<std::optional<ValueT> *>( optionalOut );
        ValueT tmp{};
        if( doExtract( row, tmp ) )
            out = std::move( tmp );
        else
            out.reset();
    }
};

// Factory: create a FieldReader for a given Arrow field + CSP struct field.
// Handles scalar types natively; list types require a registered factory (see below).
// Optional structMeta: for STRUCT columns when structField is nullptr (ColumnDispatcher path).
std::unique_ptr<FieldReader> createFieldReader(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const StructFieldPtr & structField,
    const std::shared_ptr<const StructMeta> & structMeta = nullptr
);

// Factory type for creating FieldReaders for list/array columns.
// Registered by the Python-aware layer which provides numpy-backed readers.
using ListFieldReaderFactory = std::function<
    std::unique_ptr<FieldReader>( const std::shared_ptr<::arrow::Field> &, const StructFieldPtr & )>;

// Register a factory for creating list field readers.
void registerListFieldReaderFactory( ListFieldReaderFactory factory );

}

#endif
