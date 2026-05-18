// Arrow type visitor: maps arrow::Type::type to the corresponding C++ value type.
// Eliminates repeated switch statements on arrow types across the codebase.
//
// Usage:
//   visitArrowValueType( typeId,
//       [&]( auto tag ) -> ReturnType {
//           using T = typename decltype( tag )::type;
//           return doSomething<T>( ... );
//       },
//       [&]() -> ReturnType { /* unsupported type fallback */ } );

#ifndef _IN_CSP_ADAPTERS_ARROW_ArrowTypeVisitor_H
#define _IN_CSP_ADAPTERS_ARROW_ArrowTypeVisitor_H

#include <csp/core/Exception.h>
#include <csp/core/Time.h>
#include <csp/engine/DialectGenericType.h>
#include <csp/engine/Struct.h>
#include <arrow/type.h>

namespace csp::adapters::arrow
{

template<typename T>
struct TypeTag { using type = T; };

// Invokes fn( TypeTag<CppType>{} ) for the C++ value type corresponding to
// the given arrow type.  Calls onDefault() for unrecognised arrow types.
template<typename Fn, typename DefaultFn>
decltype(auto) visitArrowValueType( ::arrow::Type::type typeId, Fn && fn, DefaultFn && onDefault )
{
    switch( typeId )
    {
        // --- Numeric ---
        case ::arrow::Type::BOOL:   return fn( TypeTag<bool>{} );
        case ::arrow::Type::INT8:   return fn( TypeTag<int8_t>{} );
        case ::arrow::Type::INT16:  return fn( TypeTag<int16_t>{} );
        case ::arrow::Type::INT32:  return fn( TypeTag<int32_t>{} );
        case ::arrow::Type::INT64:  return fn( TypeTag<int64_t>{} );
        case ::arrow::Type::UINT8:  return fn( TypeTag<uint8_t>{} );
        case ::arrow::Type::UINT16: return fn( TypeTag<uint16_t>{} );
        case ::arrow::Type::UINT32: return fn( TypeTag<uint32_t>{} );
        case ::arrow::Type::UINT64: return fn( TypeTag<uint64_t>{} );
        case ::arrow::Type::HALF_FLOAT:
        case ::arrow::Type::FLOAT:
        case ::arrow::Type::DOUBLE: return fn( TypeTag<double>{} );

        // --- String / Binary ---
        case ::arrow::Type::STRING:
        case ::arrow::Type::LARGE_STRING:
        case ::arrow::Type::BINARY:
        case ::arrow::Type::LARGE_BINARY:
        case ::arrow::Type::FIXED_SIZE_BINARY:
        case ::arrow::Type::DICTIONARY:
            return fn( TypeTag<std::string>{} );

        // --- Temporal ---
        case ::arrow::Type::TIMESTAMP: return fn( TypeTag<DateTime>{} );
        case ::arrow::Type::DURATION:  return fn( TypeTag<TimeDelta>{} );
        case ::arrow::Type::DATE32:
        case ::arrow::Type::DATE64:    return fn( TypeTag<Date>{} );
        case ::arrow::Type::TIME32:
        case ::arrow::Type::TIME64:    return fn( TypeTag<Time>{} );

        // --- List / Struct ---
        case ::arrow::Type::LIST:
        case ::arrow::Type::LARGE_LIST:
            return fn( TypeTag<DialectGenericType>{} );
        case ::arrow::Type::STRUCT:
            return fn( TypeTag<StructPtr>{} );

        default:
            return onDefault();
    }
}

// Overload that throws TypeError for unrecognised arrow types.
template<typename Fn>
decltype(auto) visitArrowValueType( ::arrow::Type::type typeId, Fn && fn )
{
    return visitArrowValueType( typeId, std::forward<Fn>( fn ),
        [typeId]() -> decltype( fn( TypeTag<bool>{} ) )
        {
            CSP_THROW( TypeError, "Unsupported arrow type id " << static_cast<int>( typeId ) );
        } );
}

} // namespace csp::adapters::arrow

#endif
