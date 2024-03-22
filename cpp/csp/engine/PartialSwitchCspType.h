#ifndef _IN_CSP_ENGINE_PartialSwitchCspType_H
#define _IN_CSP_ENGINE_PartialSwitchCspType_H

#include <csp/core/Exception.h>
#include <csp/core/System.h>
#include <csp/core/Time.h>
#include <csp/engine/CspType.h>
#include <stdint.h>
#include <unordered_set>
#include <type_traits>
#include <string>

namespace csp
{
CSP_DECLARE_EXCEPTION( UnsupportedSwitchType, TypeError );

template< csp::CspType::Type::_enum ...Vs >
struct PartialSwitchCspType
{
    static constexpr csp::CspType::Type::_enum V1_T = csp::CspType::Type::UNKNOWN;

    template< csp::CspType::Type::_enum ...ExtraVs >
    using Extend=PartialSwitchCspType<Vs..., ExtraVs...>;

    template< typename csp::CspType::Type::_enum T >
    static constexpr bool isSupportedType()
    {
        return false;
    }

    template< typename ArraySubtypeSwitch=void, typename F=void >
    static auto invoke( const CspType *type, F &&f )
    {
        CSP_THROW( UnsupportedSwitchType, "Unsupported type " << type -> type());
    }

    template< typename ArraySubtypeSwitch=void, typename F=void >
    static auto invoke( const CspTypePtr& type, F &&f )
    {
        return invoke( type.get(), std::forward<F>( f ) );
    }
};

using ArraySubTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT8,
        csp::CspType::Type::UINT8, csp::CspType::Type::INT16, csp::CspType::Type::UINT16, csp::CspType::Type::INT32,
        csp::CspType::Type::UINT32, csp::CspType::Type::INT64, csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE,
        csp::CspType::Type::DATETIME, csp::CspType::Type::TIMEDELTA, csp::CspType::Type::DATE, csp::CspType::Type::TIME,
        csp::CspType::Type::ENUM,
        csp::CspType::Type::STRING, csp::CspType::Type::STRUCT, csp::CspType::Type::DIALECT_GENERIC>;

template< csp::CspType::Type::_enum V1, csp::CspType::Type::_enum U1 >
struct SwitchCTypeResolver
{
    using TagType = CspType::Type::toCType<V1>;
};

template< csp::CspType::Type::_enum U1 >
struct SwitchCTypeResolver<csp::CspType::Type::ARRAY, U1>
{
    using TagType = CspType::Type::toCArrayType<typename CspType::Type::toCType<U1>::type>;
};

template< csp::CspType::Type::_enum V1, csp::CspType::Type::_enum ...Vs >
struct PartialSwitchCspType<V1, Vs...>
{
    template< csp::CspType::Type::_enum ...ExtraVs >
    using Extend=PartialSwitchCspType<Vs..., ExtraVs...>;
private:
    template< typename F >
    static F makeF();

    template< typename ArraySubSwitchType >
    using TagType = typename SwitchCTypeResolver<V1, ArraySubSwitchType::V1_T>::TagType;

    template< typename F, typename ArraySubSwitchType >
    using ResolveReturnType = decltype( makeF<F>()( TagType<ArraySubSwitchType>()));
public:
    static constexpr csp::CspType::Type::_enum V1_T = V1;

    template< typename csp::CspType::Type::_enum T >
    static constexpr bool isSupportedType()
    {
        return T == V1 || PartialSwitchCspType<Vs...>::template isSupportedType<T>();
    }

    template< typename ArraySubTypeSwitchT=ArraySubTypeSwitch, typename F=void >
    static auto invoke( const CspType *type, F &&f )
    {
        using R_T = ResolveReturnType<F, ArraySubTypeSwitchT>;
        switch( type -> type())
        {
            case CspType::Type::BOOL:
                return handleType<CspType::Type::BOOL, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::INT8:
                return handleType<CspType::Type::INT8, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::UINT8:
                return handleType<CspType::Type::UINT8, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::INT16:
                return handleType<CspType::Type::INT16, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::UINT16:
                return handleType<CspType::Type::UINT16, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::INT32:
                return handleType<CspType::Type::INT32, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::UINT32:
                return handleType<CspType::Type::UINT32, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::INT64:
                return handleType<CspType::Type::INT64, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::UINT64:
                return handleType<CspType::Type::UINT64, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::DOUBLE:
                return handleType<CspType::Type::DOUBLE, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::DATETIME:
                return handleType<CspType::Type::DATETIME, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::TIMEDELTA:
                return handleType<CspType::Type::TIMEDELTA, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::DATE:
                return handleType<CspType::Type::DATE, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::TIME:
                return handleType<CspType::Type::TIME, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::ENUM:
                return handleType<CspType::Type::ENUM, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::STRING:
                return handleType<CspType::Type::STRING, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::STRUCT:
                return handleType<CspType::Type::STRUCT, F, R_T>( type, std::forward<F>( f ));
            case CspType::Type::ARRAY:
                return handleArrayType<F, R_T, ArraySubTypeSwitchT>( type, std::forward<F>( f ));
            case CspType::Type::DIALECT_GENERIC:
                return handleType<CspType::Type::DIALECT_GENERIC, F, R_T>( type, std::forward<F>( f ));
            case csp::CspType::Type::UNKNOWN:
            case csp::CspType::Type::NUM_TYPES:
                CSP_THROW( TypeError, "Unexpected CspType: " << type -> type());
        }
        CSP_THROW( TypeError, "Unexpected CspType: " << type -> type());
    }

    template< typename ArraySubtypeSwitch=void, typename F=void >
    static auto invoke( const CspTypePtr& type, F &&f )
    {
        return invoke( type.get(), std::forward<F>( f ) );
    }

private:
    template< typename F,
            typename R_T,
            typename ArraySubTypeSwitchT,
            csp::CspType::Type::_enum T = CspType::Type::ARRAY,
            std::enable_if_t<isSupportedType<T>(), void *> = nullptr >
    static R_T handleArrayType( const CspType *type, F &&f )
    {
        const auto *arrayType = static_cast<const CspArrayType *>( type );

        return ArraySubTypeSwitchT::invoke( arrayType -> elemType().get(), [ &f ]( auto tag )
        {
            return f( CspType::Type::toCArrayType<typename decltype(tag)::type>());
        } );
    }

    template< typename F,
            typename R_T,
            typename ArraySubTypeSwitchT,
            csp::CspType::Type::_enum T = CspType::Type::ARRAY,
            std::enable_if_t<!isSupportedType<T>(), void *> = nullptr >
    static R_T handleArrayType( const CspType *type, F &&f )
    {
        CSP_THROW( UnsupportedSwitchType, "Unsupported type " << CspType::Type( T ));
    }

    template< csp::CspType::Type::_enum T,
            typename F,
            typename R_T,
            std::enable_if_t<isSupportedType<T>(), void *> = nullptr >
    static R_T handleType( const CspType *type, F &&f )
    {
        return f( CspType::Type::toCType<T>());
    }

    template< csp::CspType::Type::_enum T,
            typename F,
            typename R_T,
            std::enable_if_t<!isSupportedType<T>(), void *> = nullptr >
    static R_T handleType( const CspType *type, F &&f )
    {
        CSP_THROW( UnsupportedSwitchType, "Unsupported type " << CspType::Type( T ));
    }
};

template< typename T1, typename T2 >
struct CspTypeSwitchConcat
{
};

template< typename T1, csp::CspType::Type::_enum ...Vs2 >
struct CspTypeSwitchConcat<T1, PartialSwitchCspType<Vs2...>>
{
private:
    template< csp::CspType::Type::_enum ...Vs1 >
    static PartialSwitchCspType<Vs1..., Vs2...> aux( PartialSwitchCspType<Vs1...> );

public:
    using type = decltype( aux( T1()));
};

using AllCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT8,
        csp::CspType::Type::UINT8, csp::CspType::Type::INT16, csp::CspType::Type::UINT16, csp::CspType::Type::INT32,
        csp::CspType::Type::UINT32, csp::CspType::Type::INT64, csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE,
        csp::CspType::Type::DATETIME, csp::CspType::Type::TIMEDELTA, csp::CspType::Type::DATE, csp::CspType::Type::TIME, csp::CspType::Type::ENUM,
        csp::CspType::Type::STRING, csp::CspType::Type::STRUCT, csp::CspType::Type::ARRAY, csp::CspType::Type::DIALECT_GENERIC>;
using ArithmeticCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT8,
        csp::CspType::Type::UINT8, csp::CspType::Type::INT16, csp::CspType::Type::UINT16, csp::CspType::Type::INT32,
        csp::CspType::Type::UINT32, csp::CspType::Type::INT64, csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE>;
using NativeCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT8,
        csp::CspType::Type::UINT8, csp::CspType::Type::INT16, csp::CspType::Type::UINT16, csp::CspType::Type::INT32,
        csp::CspType::Type::UINT32, csp::CspType::Type::INT64, csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE,
        csp::CspType::Type::DATETIME, csp::CspType::Type::TIMEDELTA, csp::CspType::Type::DATE, csp::CspType::Type::TIME, csp::CspType::Type::ENUM>;
using PrimitiveCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT8,
        csp::CspType::Type::UINT8, csp::CspType::Type::INT16, csp::CspType::Type::UINT16, csp::CspType::Type::INT32,
        csp::CspType::Type::UINT32, csp::CspType::Type::INT64, csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE,
        csp::CspType::Type::DATETIME, csp::CspType::Type::TIMEDELTA, csp::CspType::Type::DATE, csp::CspType::Type::TIME, csp::CspType::Type::ENUM, csp::CspType::Type::STRING>;


/**
 * A class that that defines a single member type "type" to PartialSwitchCspType of all types that can be constructed from T.
 * For example ConstructibleTypeSwitchAux<double> will have its type defined to
 * PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::DOUBLE> since double can be cast to bool or double.
 * @tparam T
 */
template< typename T >
struct ConstructibleTypeSwitchAux
{
};

template<>
struct ConstructibleTypeSwitchAux<bool>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT8, csp::CspType::Type::UINT8, csp::CspType::Type::INT16,
            csp::CspType::Type::UINT16, csp::CspType::Type::INT32, csp::CspType::Type::UINT32, csp::CspType::Type::INT64,
            csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<std::int8_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT8, csp::CspType::Type::INT16,
            csp::CspType::Type::INT32, csp::CspType::Type::INT64, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<std::uint8_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::UINT8, csp::CspType::Type::INT16,
            csp::CspType::Type::UINT16, csp::CspType::Type::INT32, csp::CspType::Type::UINT32, csp::CspType::Type::INT64,
            csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<std::int16_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT16,
            csp::CspType::Type::INT32, csp::CspType::Type::INT64, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<std::uint16_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL,
            csp::CspType::Type::UINT16, csp::CspType::Type::INT32, csp::CspType::Type::UINT32, csp::CspType::Type::INT64,
            csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE>;
};
template<>
struct ConstructibleTypeSwitchAux<std::int32_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT32, csp::CspType::Type::INT64, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<std::uint32_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::UINT32, csp::CspType::Type::INT64,
            csp::CspType::Type::UINT64, csp::CspType::Type::DOUBLE>;
};
template<>
struct ConstructibleTypeSwitchAux<std::int64_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::INT64, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<std::uint64_t>
{
    // Note we allow INT64 but it should be range checked
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::UINT64, csp::CspType::Type::INT64, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<std::double_t>
{
    using type = PartialSwitchCspType<csp::CspType::Type::BOOL, csp::CspType::Type::DOUBLE>;
};

template<>
struct ConstructibleTypeSwitchAux<csp::DateTime>
{
    using type = PartialSwitchCspType<csp::CspType::Type::DATETIME>;
};

template<>
struct ConstructibleTypeSwitchAux<csp::TimeDelta>
{
    using type = PartialSwitchCspType<csp::CspType::Type::TIMEDELTA>;
};

template<>
struct ConstructibleTypeSwitchAux<csp::Date>
{
    using type = PartialSwitchCspType<csp::CspType::Type::DATE>;
};

template<>
struct ConstructibleTypeSwitchAux<csp::Time>
{
    using type = PartialSwitchCspType<csp::CspType::Type::TIME>;
};

template<>
struct ConstructibleTypeSwitchAux<csp::CspEnum>
{
    using type = PartialSwitchCspType<csp::CspType::Type::ENUM>;
};

template<>
struct ConstructibleTypeSwitchAux<std::string>
{
    using type = PartialSwitchCspType<csp::CspType::Type::STRING>;
};

template< typename StorageT >
struct ConstructibleTypeSwitchAux<std::vector<StorageT>>
{
    using type = PartialSwitchCspType<csp::CspType::Type::ARRAY>;
};

template<>
struct ConstructibleTypeSwitchAux<StructPtr>
{
    using type = PartialSwitchCspType<csp::CspType::Type::STRUCT>;
};

template<>
struct ConstructibleTypeSwitchAux<const StructPtr>
{
    using type = PartialSwitchCspType<csp::CspType::Type::STRUCT>;
};

template<>
struct ConstructibleTypeSwitchAux<DialectGenericType>
{
    using type = PartialSwitchCspType<csp::CspType::Type::DIALECT_GENERIC>;
};


template< typename T >
using ConstructibleTypeSwitch = typename ConstructibleTypeSwitchAux<T>::type;

template< typename ArraySubTypeSwitchT=ArraySubTypeSwitch, typename F=void >
inline auto switchCspType( const CspType *type, F &&f )
{
    return AllCspTypeSwitch::invoke(type, std::forward<F>(f));
}

template< typename ArraySubTypeSwitchT=ArraySubTypeSwitch, typename F=void >
inline auto switchCspType( const CspTypePtr& type, F &&f )
{
    return AllCspTypeSwitch::invoke(type, std::forward<F>(f));
}

}
#endif
