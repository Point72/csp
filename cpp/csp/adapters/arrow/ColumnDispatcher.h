// Type-erased column dispatcher: wraps FieldReader + value storage + ValueDispatcher.
//
// Each column in a RecordBatch gets a ColumnDispatcher that:
//   1. Reads a value at a given row (readValueAt)
//   2. Dispatches the value to subscribers (dispatchValue)
//   3. Supports optional symbol filtering (addSubscriber)

#ifndef _IN_CSP_ADAPTERS_ARROW_ColumnDispatcher_H
#define _IN_CSP_ADAPTERS_ARROW_ColumnDispatcher_H

#include <csp/adapters/arrow/ArrowFieldReader.h>
#include <csp/adapters/utils/ValueDispatcher.h>
#include <csp/engine/Struct.h>
#include <arrow/type.h>
#include <memory>
#include <optional>
#include <string>

namespace csp { class ManagedSimInputAdapter; }

namespace csp::adapters::arrow
{

class ColumnDispatcher
{
public:
    virtual ~ColumnDispatcher() = default;

    // Dispatch current value to subscribers.
    virtual void dispatchValue( const utils::Symbol * symbol ) = 0;

    virtual void addSubscriber( ManagedSimInputAdapter * adapter,
                                std::optional<utils::Symbol> symbol ) = 0;

    // Access current value as std::optional<T>.
    template<typename T>
    std::optional<T> & getCurValue()
    {
        return *static_cast<std::optional<T> *>( getCurValueUntyped() );
    }

    // Read value at row into internal storage without advancing cursors.
    void readValueAt( int64_t row )
    {
        m_fieldReader -> readValueAt( row, getCurValueUntyped() );
    }

    void bindColumn( const ::arrow::Array * column ) { m_fieldReader -> bindColumn( column ); }

    ::arrow::Type::type arrowTypeId() const { return m_arrowTypeId; }

protected:
    ColumnDispatcher( std::string name, std::unique_ptr<FieldReader> reader,
                      ::arrow::Type::type arrowTypeId )
        : m_columnName( std::move( name ) ), m_fieldReader( std::move( reader ) ),
          m_arrowTypeId( arrowTypeId )
    {
    }

    virtual void * getCurValueUntyped() = 0;

    std::string                  m_columnName;
    std::unique_ptr<FieldReader> m_fieldReader;
    ::arrow::Type::type          m_arrowTypeId;
};

// Factory: create a typed ColumnDispatcher for an arrow field.
// Returns nullptr for unsupported types or STRUCT without structMeta.
std::unique_ptr<ColumnDispatcher> createColumnDispatcher(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const std::shared_ptr<const StructMeta> & structMeta = nullptr );

} // namespace csp::adapters::arrow

#endif // _IN_CSP_ADAPTERS_ARROW_ColumnDispatcher_H
