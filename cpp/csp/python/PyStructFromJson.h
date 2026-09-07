#pragma once

#include <csp/python/Conversions.h>
#include <rapidjson/document.h>

namespace csp::python {
    
    StructPtr structFromJson( const StructMetaPtr& struct_meta, const rapidjson::Value& jValue );

    template<typename T>
    inline T fromJson( const rapidjson::Value & jValue )
    {
        static_assert( !std::is_same<T,T>::value, "no fromJson method implemented for type" );
        return T{};
    }

    template<typename T>
    struct FromJson
    {
        static T impl( const rapidjson::Value & jValue, const CspType & type )
        {
            return fromJson<T>( jValue );
        }
    };

    template<typename T>
    inline T fromJson( const rapidjson::Value & jValue, const CspType & type )
    {
        return FromJson<T>::impl( jValue, type );
    }

    template<typename StorageT>
    struct FromJson<std::vector<StorageT>>
    {
        static std::vector<StorageT> impl( const rapidjson::Value& jValue, const CspType& arrayType ) {
            if (!jValue.IsArray()) {
                CSP_THROW(TypeError, "expected json array.");
            }
            using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;
            const CspType& elemType = *static_cast<const CspArrayType &>( arrayType ).elemType();

            std::vector<StorageT> out;
            out.reserve( jValue.Size() );

            for (auto it = jValue.Begin(); it != jValue.End(); ++it) {
                out.emplace_back( fromJson<ElemT>(*it, elemType));
            }

            return out;
        }
    };

    //nested structs need the CspType to reach their own meta
    template<>
    struct FromJson<StructPtr>
    {
        static StructPtr impl( const rapidjson::Value & jValue, const CspType & type )
        {
            return structFromJson( static_cast<const CspStructType &>( type ).meta(), jValue );
        }
    };

    //to_json writes enums as their name
    template<>
    struct FromJson<CspEnum>
    {
        static CspEnum impl( const rapidjson::Value & jValue, const CspType & type )
        {
            if( !jValue.IsString() )
                CSP_THROW( TypeError, "expected an enum name string in json" );
            return static_cast<const CspEnumType &>( type ).meta() -> fromString( jValue.GetString() );
        }
    };

    //json carries no python objects, so generic fields take the python from_json path
    template<>
    struct FromJson<DialectGenericType>
    {
        static DialectGenericType impl( const rapidjson::Value & jValue, const CspType & type )
        {
            CSP_THROW( TypeError, "from_json does not support generic object fields" );
            return DialectGenericType();
        }
    };

    //the date/time formats below mimic the sprintf calls in PyStructToJson.cpp.

    //"%04u-%02u-%02u"
    template<>
    inline Date fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsString() ) CSP_THROW( TypeError, "expected a date string in json" );
        unsigned year, month, day;
        if( sscanf( jValue.GetString(), "%4u-%2u-%2u", &year, &month, &day ) != 3 )
            CSP_THROW( ValueError, "malformed date in json: " << jValue.GetString() );
        return Date( year, month, day );
    }

    //%02u:%02u:%02u.%06u
    template<>
    inline Time fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsString() ) CSP_THROW( TypeError, "expected a time string in json" );
        unsigned hour, minute, second, micros;
        if( sscanf( jValue.GetString(), "%2u:%2u:%2u.%6u", &hour, &minute, &second, &micros ) != 4 )
            CSP_THROW( ValueError, "malformed time in json: " << jValue.GetString() );
        return Time( hour, minute, second, micros * NANOS_PER_MICROSECOND );
    }

    //%04u-%02u-%02uT%02u:%02u:%02u.%06u+00:00, csp=utc so not parsing offset
    template<>
    inline DateTime fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsString() ) CSP_THROW( TypeError, "expected a datetime string in json" );
        unsigned year, month, day, hour, minute, second, micros;
        if( sscanf( jValue.GetString(), "%4u-%2u-%2uT%2u:%2u:%2u.%6u",
                    &year, &month, &day, &hour, &minute, &second, &micros ) != 7 )
            CSP_THROW( ValueError, "malformed datetime in json: " << jValue.GetString() );
        return DateTime( year, month, day, hour, minute, second, micros * NANOS_PER_MICROSECOND );
    }

    //<sign><seconds>.<micros>
    template<>
    inline TimeDelta fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsString() ) CSP_THROW( TypeError, "expected a timedelta string in json" );
        const char * str = jValue.GetString();
        bool negative = ( *str == '-' );
        if( negative || *str == '+' )
            ++str;

        uint64_t seconds;
        unsigned micros;
        if( sscanf( str, "%lu.%6u", &seconds, &micros ) != 2 )
            CSP_THROW( ValueError, "malformed timedelta in json: " << jValue.GetString() );

        int64_t nanos = seconds * NANOS_PER_SECOND + micros * NANOS_PER_MICROSECOND;
        return TimeDelta::fromNanoseconds( negative ? -nanos : nanos );
    }

    template<>
    inline int64_t fromJson( const rapidjson::Value & jValue) {
        if (!jValue.IsInt64()) {
            CSP_THROW(TypeError, "Expected int64 in JSON.");
        }
        return jValue.GetInt64();
    }

    //narrow int widths go via int64 then a range check.
    template<typename T>
    inline T narrowIntFromJson( const rapidjson::Value & jValue )
    {
        int64_t v = fromJson<int64_t>( jValue );
        if( v < static_cast<int64_t>( std::numeric_limits<T>::min() ) ||
            v > static_cast<int64_t>( std::numeric_limits<T>::max() ) )
            CSP_THROW( ValueError, "json value " << v << " is out of range for the field type" );
        return static_cast<T>( v );
    }

    template<> inline int8_t   fromJson( const rapidjson::Value & jValue ) { return narrowIntFromJson<int8_t>( jValue ); }
    template<> inline int16_t  fromJson( const rapidjson::Value & jValue ) { return narrowIntFromJson<int16_t>( jValue ); }
    template<> inline int32_t  fromJson( const rapidjson::Value & jValue ) { return narrowIntFromJson<int32_t>( jValue ); }
    template<> inline uint8_t  fromJson( const rapidjson::Value & jValue ) { return narrowIntFromJson<uint8_t>( jValue ); }
    template<> inline uint16_t fromJson( const rapidjson::Value & jValue ) { return narrowIntFromJson<uint16_t>( jValue ); }
    template<> inline uint32_t fromJson( const rapidjson::Value & jValue ) { return narrowIntFromJson<uint32_t>( jValue ); }

    template<>
    inline uint64_t fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsUint64() ) CSP_THROW( TypeError, "expected uint64 in json" );
        return jValue.GetUint64();
    }

    template<> 
    inline bool fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsBool() ) CSP_THROW( TypeError, "expected bool in json" );
        return jValue.GetBool();
    }

    template<> 
    inline double fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsNumber()) CSP_THROW( TypeError, "expected double in json" );
        return jValue.GetDouble();
    }

    template<> 
    inline std::string fromJson( const rapidjson::Value & jValue )
    {
        if( !jValue.IsString() ) CSP_THROW( TypeError, "expected string in json" );
        return jValue.GetString();
    }

}
