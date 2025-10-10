#include <csp/adapters/utils/JSONMessageStructConverter.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <type_traits>

namespace csp::adapters::utils
{

using SupportedCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL,
                                                    csp::CspType::Type::INT32,
                                                    csp::CspType::Type::INT64,
                                                    csp::CspType::Type::DOUBLE,
                                                    csp::CspType::Type::DATE,
                                                    csp::CspType::Type::DATETIME,
                                                    csp::CspType::Type::STRING,
                                                    csp::CspType::Type::ENUM,
                                                    csp::CspType::Type::STRUCT,
                                                    csp::CspType::Type::ARRAY
                                                    >;

using SupportedArrayCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL,
                                                    csp::CspType::Type::INT32,
                                                    csp::CspType::Type::INT64,
                                                    csp::CspType::Type::DOUBLE,
                                                    csp::CspType::Type::DATETIME,
                                                    csp::CspType::Type::STRING,
                                                    csp::CspType::Type::ENUM
                                                    >;


template<>
bool JSONMessageStructConverter::convertJSON( const char * fieldname, const rapidjson::Value & jValue, bool * )
{
    if( jValue.IsBool() )
        return jValue.GetBool();
    else
        CSP_THROW( TypeError, "expected type BOOL for json field " << fieldname );
}

template<>
int32_t JSONMessageStructConverter::convertJSON( const char * fieldname, const rapidjson::Value & jValue, int32_t * )
{
    if( jValue.IsInt() )
        return jValue.GetInt();
    else
        CSP_THROW( TypeError, "expected INT32 type for json field " << fieldname );
}

template<>
int64_t JSONMessageStructConverter::convertJSON( const char * fieldname, const rapidjson::Value & jValue, int64_t * )
{
    if( jValue.IsInt64() )
        return jValue.GetInt64();
    else
        CSP_THROW( TypeError, "expected INT64 type for json field " << fieldname );
}

template<>
double JSONMessageStructConverter::convertJSON( const char * fieldname, const rapidjson::Value & jValue, double * )
{
    if( jValue.IsNumber() )
        return jValue.GetDouble();
    else
        CSP_THROW( TypeError, "expected DOUBLE type for json field " << fieldname );
}

template<>
std::string JSONMessageStructConverter::convertJSON( const char * fieldname, const rapidjson::Value & jValue, std::string * )
{
    if( jValue.IsString() )
        return jValue.GetString();
    else
        CSP_THROW( TypeError, "expected STRING type for json field " << fieldname );
}

template<>
Date JSONMessageStructConverter::convertJSON( const char * fieldname, const rapidjson::Value & jValue, Date * )
{
    if( jValue.IsString() )
        return Date::fromYYYYMMDD( jValue.GetString() );
    else
        CSP_THROW( TypeError, "expected STRING type for json DATE field " << fieldname );
}

template<>
DateTime JSONMessageStructConverter::convertJSON( const char * fieldname, const rapidjson::Value & jValue, DateTime * )
{
    if( jValue.IsUint64() )
    {
        uint64_t raw = jValue.GetUint64();
        DateTime dt;
        switch( m_datetimeType )
        {
            case DateTimeWireType::UINT64_NANOS:   dt = DateTime::fromNanoseconds( raw ); break;
            case DateTimeWireType::UINT64_MICROS:  dt = DateTime::fromMicroseconds( raw ); break;
            case DateTimeWireType::UINT64_MILLIS:  dt = DateTime::fromMilliseconds( raw ); break;
            case DateTimeWireType::UINT64_SECONDS: dt = DateTime::fromSeconds( raw ); break;

            case DateTimeWireType::UNKNOWN:
            case DateTimeWireType::NUM_TYPES:
                CSP_THROW( TypeError, "json field " << fieldname << " is datetime but datetimeType is not configured" );
        }

        return dt;
    }
    else
        CSP_THROW( TypeError, "expected UINT64 for DATETIME for json field " << fieldname );
}

template<>
CspEnum JSONMessageStructConverter::convertJSON( const char * fieldname, const CspType & type, const FieldEntry &, const rapidjson::Value & jValue, CspEnum * )
{
    if( !jValue.IsString() )
        CSP_THROW( TypeError, "expected ENUM type for json field " << fieldname );

    auto & cspEnumType = static_cast<const CspEnumType &>( type );
    return cspEnumType.meta() -> fromString( jValue.GetString() );
}

template<>
StructPtr JSONMessageStructConverter::convertJSON( const char * fieldname, const CspType &, const FieldEntry & entry, const rapidjson::Value & jValue, StructPtr * )
{
    if( !jValue.IsObject() )
        CSP_THROW( TypeError, "expected Nested object type for json field " << fieldname );

    const CspStructType & sType = static_cast<const CspStructType &>( *entry.sField -> type() );
    auto struct_ = sType.meta() -> create();
    auto & fields = *entry.nestedFields;
    for( auto jit = jValue.MemberBegin(); jit != jValue.MemberEnd(); ++jit )
    {
        auto sIt = fields.find( jit -> name.GetString() );
        if( sIt == fields.end() )
            continue;

        auto & nestedEntry = sIt -> second;

        SupportedCspTypeSwitch::invoke<SupportedArrayCspTypeSwitch>(
            nestedEntry.sField -> type().get(), [this,&jit,&nestedEntry,&struct_]( auto tag )
            {
                using T = typename decltype(tag)::type;
                auto & jValue = jit -> value;

                nestedEntry.sField -> setValue( struct_.get(), convertJSON( jit -> name.GetString(), *nestedEntry.sField -> type(), nestedEntry, jValue, static_cast<T*>( nullptr ) ) );
            } );
    }

    return struct_;
}

template<typename StorageT>
std::vector<StorageT> JSONMessageStructConverter::convertJSON( const char * fieldname, const CspType & type, const FieldEntry &, const rapidjson::Value & jValue, std::vector<StorageT> * x )
{
    if( !jValue.IsArray() )
        CSP_THROW( TypeError, "expected ARRAY type for json field " << fieldname );

    auto jArray = jValue.GetArray();

    const CspType & elemType = *static_cast<const CspArrayType &>( type ).elemType();

    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;
    std::vector<StorageT> out;
    out.reserve( jArray.Size() );
    for( auto & v : jArray )
    {
        //note that we dont pass FieldEntry to convert here, this doesnt support arrays of structs
        out.emplace_back( convertJSON( fieldname, elemType, {}, v, ( ElemT * ) nullptr) );
    }

    return out;
}

JSONMessageStructConverter::JSONMessageStructConverter( const CspTypePtr & type,
                                                        const Dictionary & properties ) : MessageStructConverter( type, properties )
{
    if( type -> type() != CspType::Type::STRUCT )
        CSP_THROW( TypeError, "JSONMessageStructConverter expects type struct got " << type -> type() );

    const Dictionary & fieldMap = *properties.get<DictionaryPtr>( "field_map" );
    m_datetimeType = DateTimeWireType( properties.get<std::string>( "datetime_type" ) );
    m_fields = buildFields( static_cast<const CspStructType &>( *type ), fieldMap );
}

JSONMessageStructConverter::Fields JSONMessageStructConverter::buildFields( const CspStructType & type, const Dictionary & fieldMap )
{
    JSONMessageStructConverter::Fields out;

    for( auto it = fieldMap.begin(); it != fieldMap.end(); ++it )
    {
        auto & fieldName  = it.key();

        std::string structField;
        DictionaryPtr nestedFieldMap;

        if( it.hasValue<std::string>() )
            structField = it.value<std::string>();
        else
        {
            if( !it.hasValue<DictionaryPtr>() )
                CSP_THROW( TypeError, "fieldMap expected string or dict for field " << fieldName << " on struct " << type.meta() -> name() );
            auto nestedDict = it.value<DictionaryPtr>();
            if( nestedDict -> size() != 1 )
                CSP_THROW( ValueError, "Expected nested fieldmap for incoming JSON field " << fieldName << " to have a single key : map entry" );
            structField    = nestedDict -> begin().key();
            nestedFieldMap = nestedDict -> begin().value<DictionaryPtr>();
        }

        auto sField = type.meta() -> field( structField );
        if( !sField )
            CSP_THROW( ValueError, "field " << structField << " is not a valid field on struct type " << type.meta() -> name() );

        std::shared_ptr<Fields> nestedFields;
        if( sField -> type() -> type() == CspType::Type::STRUCT )
        {
            if( !nestedFieldMap )
                CSP_THROW( ValueError, "invalid field_map entry for nested struct field " << sField -> fieldname() << " on struct type " << type.meta() -> name() );
            nestedFields = std::make_shared<Fields>( buildFields( static_cast<const CspStructType &>( *sField -> type() ), *nestedFieldMap ) );
        }

        //keep strings around to keep const char * keys alive
        m_jsonkeys.emplace_back( fieldName );
        out.emplace( m_jsonkeys.back().c_str(), FieldEntry{ sField, nestedFields } );
    }
    return out;
}

csp::StructPtr JSONMessageStructConverter::asStruct( void * bytes, size_t size )
{
    const char * rawmsg = (const char *) bytes;

    rapidjson::Document document;
    rapidjson::ParseResult ok = document.Parse<rapidjson::kParseNanAndInfFlag>( rawmsg, size );
    if( !ok )
        CSP_THROW( ValueError, "Failed to parse message as JSON: " << rapidjson::GetParseError_En( ok.Code() ) << " on msg: " << std::string( rawmsg, size ) );

    StructPtr data = m_structMeta -> create();

    for( auto jit = document.MemberBegin(); jit != document.MemberEnd(); ++jit )
    {
        auto sIt = m_fields.find( jit -> name.GetString() );
        if( sIt == m_fields.end() )
            continue;

        auto & entry = sIt -> second;
        SupportedCspTypeSwitch::invoke<SupportedArrayCspTypeSwitch>(
            entry.sField -> type().get(), [this,&jit,&entry,&data]( auto tag )
            {
                using T = typename decltype(tag)::type;
                auto & jValue = jit -> value;
                entry.sField -> setValue( data.get(), convertJSON( jit -> name.GetString(), *entry.sField -> type(), entry, jValue, static_cast<T*>( nullptr ) ) );
            }
        );
    }

    return data;
}
}
