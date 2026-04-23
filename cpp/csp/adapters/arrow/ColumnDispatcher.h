// Type-erased column dispatcher: wraps FieldReader + value storage + ValueDispatcher.
//
// Each column in a RecordBatch gets a ColumnDispatcher that knows how to:
//   1. Read the next row from the bound arrow array (readNextValue)
//   2. Dispatch the value to registered subscribers (dispatchValue)
//   3. Register subscribers with optional symbol filtering (addSubscriber)
//
// Concrete dispatchers are typed (TypedColumnDispatcher<T>) but the base class
// provides a uniform interface for RecordBatchRowProcessor to iterate columns
// without knowing value types.

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

    // Read next row value from bound arrow array into internal storage.
    virtual void readNextValue() = 0;

    // Dispatch current value to all registered subscribers.
    virtual void dispatchValue( const utils::Symbol * symbol ) = 0;

    // Register a subscriber for this column.
    virtual void addSubscriber( ManagedSimInputAdapter * adapter,
                                std::optional<utils::Symbol> symbol ) = 0;

    // Access current value as std::optional<T>. Caller must know the correct T.
    template<typename T>
    std::optional<T> & getCurValue()
    {
        return *static_cast<std::optional<T> *>( getCurValueUntyped() );
    }

    // Bind to a new arrow column array (resets row counter).
    void bindColumn( const ::arrow::Array * column ) { m_fieldReader -> bindColumn( column ); }

    const std::string & columnName() const { return m_columnName; }
    FieldReader & fieldReader() { return *m_fieldReader; }
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
// Returns nullptr for unsupported types, and for STRUCT when structMeta is not provided.
std::unique_ptr<ColumnDispatcher> createColumnDispatcher(
    const std::shared_ptr<::arrow::Field> & arrowField,
    const std::shared_ptr<const StructMeta> & structMeta = nullptr );

} // namespace csp::adapters::arrow

#endif
