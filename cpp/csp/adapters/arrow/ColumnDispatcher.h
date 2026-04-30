// Type-erased column dispatcher: value extraction + dispatch to subscribers.
//
// Each column in a RecordBatch gets a ColumnDispatcher that:
//   1. Reads a value at a given row (readValueAt)
//   2. Dispatches the value to subscribers (dispatchValue)
//   3. Supports optional symbol filtering (addSubscriber)
//
// For scalar types, TypedColumnDispatcher uses an InlineReader (cached typed
// pointer + extract lambda) for zero-overhead value extraction.
// For complex types (STRUCT, DICT), it uses ErasedReader wrapping a FieldReader.

#ifndef _IN_CSP_ADAPTERS_ARROW_ColumnDispatcher_H
#define _IN_CSP_ADAPTERS_ARROW_ColumnDispatcher_H

#include <csp/adapters/arrow/ArrowFieldReader.h>
#include <csp/adapters/utils/ValueDispatcher.h>
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

    virtual void dispatchValue( const utils::Symbol * symbol ) = 0;

    virtual void addSubscriber( ManagedSimInputAdapter * adapter,
                                std::optional<utils::Symbol> symbol ) = 0;

    // Access current value (used by adapter manager for time/symbol columns).
    template<typename T>
    std::optional<T> & getCurValue()
    {
        return *static_cast<std::optional<T> *>( getCurValueUntyped() );
    }

    // Read value at row into internal storage (hot path).
    virtual void readValueAt( int64_t row ) = 0;

    // Bind a new column array (called on each new RecordBatch).
    virtual void bindColumn( const ::arrow::Array * column ) = 0;

    ::arrow::Type::type arrowTypeId() const { return m_arrowTypeId; }

protected:
    ColumnDispatcher( std::string name, ::arrow::Type::type arrowTypeId )
        : m_arrowTypeId( arrowTypeId ), m_columnName( std::move( name ) )
    {
    }

    virtual void * getCurValueUntyped() = 0;

    ::arrow::Type::type m_arrowTypeId;
    std::string         m_columnName;
};

// Factory: create a typed ColumnDispatcher for an arrow field.
// Returns nullptr for unsupported types or STRUCT without structMeta.
std::unique_ptr<ColumnDispatcher> createColumnDispatcher(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const std::shared_ptr<const StructMeta> & structMeta = nullptr );

} // namespace csp::adapters::arrow

#endif // _IN_CSP_ADAPTERS_ARROW_ColumnDispatcher_H
