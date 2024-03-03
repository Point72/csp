#ifndef _IN_CSP_ENGINE_CSPTYPE_H
#define _IN_CSP_ENGINE_CSPTYPE_H

#include <csp/core/Enum.h>
#include <csp/core/Time.h>
#include <csp/engine/CspEnum.h>
#include <csp/engine/DialectGenericType.h>
#include <cstdint>
#include <memory>
#include <string>

namespace csp
{

class CspStringType;

class CspType
{
public:

    struct TypeTraits
    {
    public:
        
        enum _enum : uint8_t
        {
            UNKNOWN,
            BOOL,
            INT8,
            UINT8,
            INT16,
            UINT16,
            INT32,
            UINT32,
            INT64,
            UINT64,
            DOUBLE,
            DATETIME,
            TIMEDELTA,
            DATE,
            TIME,
            ENUM,

            //Native implied the data is memcpy-able
            MAX_NATIVE_TYPE = ENUM,

            STRING,
            STRUCT,
            ARRAY,

            //These types are currently all dialect specific, no native primitives
            DIALECT_GENERIC,

            NUM_TYPES
        };

        template< typename T >
        struct fromCType;

        template< uint8_t T, typename ELEM_T=void >
        struct toCType;

    protected:
        _enum m_value;
    };

    using Type = Enum<TypeTraits>;

    CspType( Type t ) : m_type( t ) {}

    Type type() const { return m_type; }

    using Ptr = std::shared_ptr<const CspType>;

    static Ptr & BOOL()            { static auto s_type = std::make_shared<const CspType>( Type::BOOL );            return s_type; }
    static Ptr & INT8()            { static auto s_type = std::make_shared<const CspType>( Type::INT8 );            return s_type; }
    static Ptr & UINT8()           { static auto s_type = std::make_shared<const CspType>( Type::UINT8 );           return s_type; }
    static Ptr & INT16()           { static auto s_type = std::make_shared<const CspType>( Type::INT16 );           return s_type; }
    static Ptr & UINT16()          { static auto s_type = std::make_shared<const CspType>( Type::UINT16 );          return s_type; }
    static Ptr & INT32()           { static auto s_type = std::make_shared<const CspType>( Type::INT32 );           return s_type; }
    static Ptr & UINT32()          { static auto s_type = std::make_shared<const CspType>( Type::UINT32 );          return s_type; }
    static Ptr & INT64()           { static auto s_type = std::make_shared<const CspType>( Type::INT64 );           return s_type; }
    static Ptr & UINT64()          { static auto s_type = std::make_shared<const CspType>( Type::UINT64 );          return s_type; }
    static Ptr & DOUBLE()          { static auto s_type = std::make_shared<const CspType>( Type::DOUBLE );          return s_type; }
    static Ptr & DATETIME()        { static auto s_type = std::make_shared<const CspType>( Type::DATETIME );        return s_type; }
    static Ptr & TIMEDELTA()       { static auto s_type = std::make_shared<const CspType>( Type::TIMEDELTA );       return s_type; }
    static Ptr & DATE()            { static auto s_type = std::make_shared<const CspType>( Type::DATE );            return s_type; }
    static Ptr & TIME()            { static auto s_type = std::make_shared<const CspType>( Type::TIME );            return s_type; }
    static Ptr & STRING();
    static Ptr & BYTES();
    static Ptr & DIALECT_GENERIC() { static auto s_type = std::make_shared<const CspType>( Type::DIALECT_GENERIC ); return s_type; }

    template<typename T>
    struct fromCType;

    using StringCType = std::string;

private:
    Type m_type;
};

class CspStringType:public CspType
{
public:
    CspStringType(bool isBytes)
    : CspType(CspType::Type::STRING), m_isBytes(isBytes) {}
    inline bool isBytes() const {return m_isBytes;}
private:
    const bool m_isBytes;
};

inline CspType::Ptr & CspType::STRING() { static CspType::Ptr s_type = std::make_shared<const CspStringType>( false ); return s_type; }
inline CspType::Ptr & CspType::BYTES() { static CspType::Ptr s_type = std::make_shared<const CspStringType>( true ); return s_type; }


using CspTypePtr = CspType::Ptr;

class CspEnum;

class CspEnumMeta;

class CspEnumType : public CspType
{
public:
    CspEnumType( std::shared_ptr<CspEnumMeta> & meta ) : CspType( CspType::Type::ENUM ),
                                                         m_meta( meta )
    {}

    const std::shared_ptr<CspEnumMeta> & meta() const { return m_meta; }

private:
    std::shared_ptr<CspEnumMeta> m_meta;
};

class Struct;
template<typename T>
class TypedStructPtr;
using StructPtr = TypedStructPtr<Struct>;

class StructMeta;

class CspStructType : public CspType
{
public:
    CspStructType( const std::shared_ptr<StructMeta> & meta ) : CspType( CspType::Type::STRUCT ),
                                                                m_meta( meta )
    {}
    
    const std::shared_ptr<StructMeta> & meta() const { return m_meta; }

private:
    std::shared_ptr<StructMeta> m_meta;
};

class CspArrayType : public CspType
{
public:
    CspArrayType( CspTypePtr elemType ) : CspType( CspType::Type::ARRAY ),
                                          m_elemType( elemType )
    {}

    const CspTypePtr & elemType() const { return m_elemType; }

    //Used by BURST mode to avoid creating more instances of CspArrayTypes than needed
    //returns CspArrayType with the given elemType
    static CspTypePtr & create( const CspTypePtr & elemType );

private:
    CspTypePtr m_elemType;
};

#ifndef __clang__
template<> struct CspType::Type::fromCType<std::_Bit_reference>            { static constexpr CspType::Type type = CspType::Type::BOOL; };
#endif

template<> struct CspType::TypeTraits::fromCType<bool>                     { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::BOOL;            };
template<> struct CspType::TypeTraits::fromCType<int8_t>                   { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::INT8;            };
template<> struct CspType::TypeTraits::fromCType<uint8_t>                  { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::UINT8;           };
template<> struct CspType::TypeTraits::fromCType<int16_t>                  { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::INT16;           };
template<> struct CspType::TypeTraits::fromCType<uint16_t>                 { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::UINT16;          };
template<> struct CspType::TypeTraits::fromCType<int32_t>                  { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::INT32;           };
template<> struct CspType::TypeTraits::fromCType<uint32_t>                 { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::UINT32;          };
template<> struct CspType::TypeTraits::fromCType<int64_t>                  { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::INT64;           };
template<> struct CspType::TypeTraits::fromCType<uint64_t>                 { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::UINT64;          };
template<> struct CspType::TypeTraits::fromCType<double>                   { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::DOUBLE;          };
template<> struct CspType::TypeTraits::fromCType<DateTime>                 { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::DATETIME;        };
template<> struct CspType::TypeTraits::fromCType<TimeDelta>                { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::TIMEDELTA;       };
template<> struct CspType::TypeTraits::fromCType<Date>                     { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::DATE;            };
template<> struct CspType::TypeTraits::fromCType<Time>                     { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::TIME;            };
template<> struct CspType::TypeTraits::fromCType<CspEnum>                  { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::ENUM;            };
template<> struct CspType::TypeTraits::fromCType<CspType::StringCType>     { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::STRING;          };
template<> struct CspType::TypeTraits::fromCType<StructPtr>                { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::STRUCT;          };
template<typename T> struct CspType::TypeTraits::fromCType<TypedStructPtr<T>> { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::STRUCT;       };
template<> struct CspType::TypeTraits::fromCType<DialectGenericType>       { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::DIALECT_GENERIC; };
template<typename T> struct CspType::TypeTraits::fromCType<std::vector<T>> { static constexpr CspType::TypeTraits::_enum type = CspType::TypeTraits::ARRAY; };

template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::BOOL,void>            { using type = bool;      };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::INT8,void>            { using type = int8_t;    };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::UINT8,void>           { using type = uint8_t;   };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::INT16,void>           { using type = int16_t;   };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::UINT16,void>          { using type = uint16_t;  };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::INT32,void>           { using type = int32_t;   };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::UINT32,void>          { using type = uint32_t;  };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::INT64,void>           { using type = int64_t;   };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::UINT64,void>          { using type = uint64_t;  };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::DOUBLE,void>          { using type = double;    };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::DATETIME,void>        { using type = DateTime;  };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::TIMEDELTA,void>       { using type = TimeDelta; };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::DATE,void>            { using type = Date; };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::TIME,void>            { using type = Time; };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::ENUM,void>            { using type = CspEnum; };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::STRING,void>          { using type = CspType::StringCType; };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::STRUCT,void>          { using type = StructPtr; };
template<> struct CspType::TypeTraits::toCType<CspType::TypeTraits::DIALECT_GENERIC,void> { using type = DialectGenericType; };
template<typename T> struct CspType::TypeTraits::toCType<CspType::TypeTraits::ARRAY,T> { using type = std::vector<T>; };

template<> struct CspType::fromCType<bool>                 { static CspTypePtr & type() { return CspType::BOOL();      } };
template<> struct CspType::fromCType<int8_t>               { static CspTypePtr & type() { return CspType::INT8();      } };
template<> struct CspType::fromCType<uint8_t>              { static CspTypePtr & type() { return CspType::UINT8();     } };
template<> struct CspType::fromCType<int16_t>              { static CspTypePtr & type() { return CspType::INT16();     } };
template<> struct CspType::fromCType<uint16_t>             { static CspTypePtr & type() { return CspType::UINT16();    } };
template<> struct CspType::fromCType<int32_t>              { static CspTypePtr & type() { return CspType::INT32();     } };
template<> struct CspType::fromCType<uint32_t>             { static CspTypePtr & type() { return CspType::UINT32();    } };
template<> struct CspType::fromCType<int64_t>              { static CspTypePtr & type() { return CspType::INT64();     } };
template<> struct CspType::fromCType<uint64_t>             { static CspTypePtr & type() { return CspType::UINT64();    } };
template<> struct CspType::fromCType<double>               { static CspTypePtr & type() { return CspType::DOUBLE();    } };
template<> struct CspType::fromCType<DateTime>             { static CspTypePtr & type() { return CspType::DATETIME();  } };
template<> struct CspType::fromCType<TimeDelta>            { static CspTypePtr & type() { return CspType::TIMEDELTA(); } };
template<> struct CspType::fromCType<Date>                 { static CspTypePtr & type() { return CspType::DATE();      } };
template<> struct CspType::fromCType<Time>                 { static CspTypePtr & type() { return CspType::TIME();      } };
template<> struct CspType::fromCType<CspType::StringCType> { static CspTypePtr & type() { return CspType::STRING();    } };
}

#endif
