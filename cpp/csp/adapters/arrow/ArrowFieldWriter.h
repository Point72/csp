// Per-column writers that serialize CSP struct field values into Arrow array builders.
//
// FieldWriter is the base class.  Every writer targets exactly one struct field.
// Scalar writers use the single-column constructor with a builder; multi-column
// writers (e.g. NDArray with data + dims) use the multi-column constructor.
// The non-virtual writeNext() checks isSet and delegates to the virtual doWrite().

#ifndef _IN_CSP_ADAPTERS_ARROW_ArrowFieldWriter_H
#define _IN_CSP_ADAPTERS_ARROW_ArrowFieldWriter_H

#include <csp/engine/Struct.h>
#include <arrow/builder.h>
#include <arrow/type.h>
#include <memory>
#include <string>
#include <vector>

namespace csp::adapters::arrow
{

class FieldWriter
{
public:
    virtual ~FieldWriter() = default;

    // Pre-allocate builder capacity for numRows.
    // Default reserves on m_builder; override for multi-builder writers.
    virtual void reserve( int64_t numRows );

    // Write one struct's field value into the builder.
    // Non-virtual: checks isSet, delegates to doWrite() or appendNull().
    void writeNext( const Struct * s );

    // Columnar bulk-write: write a range of structs into the builder.
    // Default loops over writeNext(); concrete writers override with tight loops.
    virtual void writeAll( const std::vector<StructPtr> & structs, int64_t offset, int64_t count );

    // Write a null value (used by nested struct writer when parent is null).
    // Default appends null to m_builder; NestedStructWriter overrides.
    virtual void writeNull();

    // Finalize and return the built arrays.
    // Default finishes m_builder and returns a single array.
    virtual std::vector<std::shared_ptr<::arrow::Array>> finish();

    // Column names produced by this writer.
    const std::vector<std::string> & columnNames() const { return m_columnNames; }

    // Arrow data types per column.
    const std::vector<std::shared_ptr<::arrow::DataType>> & dataTypes() const { return m_dataTypes; }

    // Access the primary builder (needed by NestedStructWriter for StructBuilder construction).
    const std::shared_ptr<::arrow::ArrayBuilder> & builder() const { return m_builder; }

    // Constructor for scalar writers (single column, single struct field, one builder)
    FieldWriter( const std::string & columnName,
                 const StructFieldPtr & field,
                 std::shared_ptr<::arrow::ArrayBuilder> builder,
                 std::shared_ptr<::arrow::DataType> dataType )
        : m_field( field ),
          m_builder( std::move( builder ) ),
          m_columnNames( { columnName } ),
          m_dataTypes( { std::move( dataType ) } )
    {
    }

    // Constructor for multi-column writers (e.g. NDArray with data + dims columns)
    FieldWriter( std::vector<std::string> columnNames,
                 std::vector<std::shared_ptr<::arrow::DataType>> dataTypes,
                 const StructFieldPtr & field )
        : m_field( field ),
          m_columnNames( std::move( columnNames ) ),
          m_dataTypes( std::move( dataTypes ) )
    {
    }

protected:
    // Write the field value when it is set. Concrete scalar writers implement this.
    virtual void doWrite( const Struct * s ) = 0;

    StructFieldPtr                                      m_field;
    std::shared_ptr<::arrow::ArrayBuilder>              m_builder;
    std::vector<std::string>                            m_columnNames;
    std::vector<std::shared_ptr<::arrow::DataType>>     m_dataTypes;
};

// Return type for createFieldWriter: the writer plus its primary builder
// (the builder is needed by nested struct writer to construct StructBuilder with child builders)
struct CreatedFieldWriter
{
    std::unique_ptr<FieldWriter>               writer;
    std::shared_ptr<::arrow::ArrayBuilder>     builder;
};

// Factory: given column name + struct field, produce a FieldWriter with its builder
CreatedFieldWriter createFieldWriter(
    const std::string & columnName,
    const StructFieldPtr & structField
);

}

#endif
