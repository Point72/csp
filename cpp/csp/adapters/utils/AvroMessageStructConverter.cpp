#include <csp/adapters/utils/AvroMessageStructConverter.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <avro/Compiler.hh>
#include <avro/GenericDatum.hh>
#include <avro/Stream.hh>
#include <avro/Node.hh>
#include <type_traits>
#include <iostream>

// Implement the fmt::formatter<avro::Name>::format() method for Windows
#ifdef _MSC_VER
template<typename FormatContext>
auto fmt::formatter<avro::Name, char>::format(const avro::Name& n, FormatContext& ctx) const -> decltype(ctx.out()) {
    return fmt::format_to(ctx.out(), "{}", n.fullname());
}
#endif

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

using SupportedArrayCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::INT32,
                                                    csp::CspType::Type::INT64,
                                                    csp::CspType::Type::DOUBLE,
                                                    csp::CspType::Type::DATETIME,
                                                    csp::CspType::Type::STRING,
                                                    csp::CspType::Type::ENUM
                                                    >;

template<>
bool AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    if( datum.type() == avro::AVRO_BOOL )
        return datum.value<bool>();
    CSP_THROW( TypeError, "expected type BOOL for avro field " << fieldname );
}

template<>
int32_t AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    if( datum.type() == avro::AVRO_INT )
        return datum.value<int32_t>();
    CSP_THROW( TypeError, "expected INT type for avro field " << fieldname );
}

template<>
int64_t AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    if( datum.type() == avro::AVRO_LONG )
        return datum.value<int64_t>();
    if( datum.type() == avro::AVRO_INT )
        return static_cast<int64_t>( datum.value<int32_t>() );
    CSP_THROW( TypeError, "expected LONG type for avro field " << fieldname << " got " << datum.type() );
}

template<>
double AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    if( datum.type() == avro::AVRO_DOUBLE )
        return datum.value<double>();
    if( datum.type() == avro::AVRO_FLOAT )
        return static_cast<double>( datum.value<float>() );
    if( datum.type() == avro::AVRO_LONG )
        return static_cast<double>( datum.value<int64_t>() );
    if( datum.type() == avro::AVRO_INT )
        return static_cast<double>( datum.value<int32_t>() );
    CSP_THROW( TypeError, "expected DOUBLE type for avro field " << fieldname << " got " << datum.type() );
}

template<>
std::string AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    if( datum.type() == avro::AVRO_STRING )
        return datum.value<std::string>();
    CSP_THROW( TypeError, "expected STRING type for avro field " << fieldname );
}

template<>
Date AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    if( datum.type() == avro::AVRO_INT )
    {
        int32_t daysSinceEpoch = datum.value<int32_t>();
        return Date::fromYYYYMMDD( "1970-01-01" ) + TimeDelta::fromDays( daysSinceEpoch );
    }
    if( datum.type() == avro::AVRO_STRING )
        return Date::fromYYYYMMDD( datum.value<std::string>().c_str() );
    CSP_THROW( TypeError, "expected INT or STRING type for avro DATE field " << fieldname );
}

template<>
DateTime AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    int64_t raw = 0;
    if( datum.type() == avro::AVRO_LONG )
        raw = datum.value<int64_t>();
    else if( datum.type() == avro::AVRO_INT )
        raw = static_cast<int64_t>( datum.value<int32_t>() );
    else
        CSP_THROW( TypeError, "expected LONG for DATETIME for avro field " << fieldname );

    switch( m_datetimeType )
    {
        case DateTimeWireType::UINT64_NANOS:   return DateTime::fromNanoseconds( raw );
        case DateTimeWireType::UINT64_MICROS:  return DateTime::fromMicroseconds( raw );
        case DateTimeWireType::UINT64_MILLIS:  return DateTime::fromMilliseconds( raw );
        case DateTimeWireType::UINT64_SECONDS: return DateTime::fromSeconds( raw );
        default:
            CSP_THROW( TypeError, "avro field " << fieldname << " is datetime but datetimeType is not configured" );
    }
}

template<>
CspEnum AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    CSP_THROW( NotImplemented, "Enum extraction requires type info, use convertAvroValue instead" );
}

template<>
StructPtr AvroMessageStructConverter::extractValue( const avro::GenericDatum & datum, const char * fieldname )
{
    CSP_THROW( NotImplemented, "Struct extraction requires field info, use convertAvroValue instead" );
}

template<typename StorageT>
std::vector<StorageT> AvroMessageStructConverter::extractArray( const avro::GenericDatum & datum, const char * fieldname, const CspType & elemType )
{
    if( datum.type() != avro::AVRO_ARRAY )
        CSP_THROW( TypeError, "expected ARRAY type for avro field " << fieldname );

    const avro::GenericArray & arr = datum.value<avro::GenericArray>();
    const std::vector<avro::GenericDatum> & elements = arr.value();

    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;
    std::vector<StorageT> out;
    out.reserve( elements.size() );

    for( const auto & elem : elements )
    {
        out.emplace_back( extractValue<ElemT>( elem, fieldname ) );
    }

    return out;
}

void AvroMessageStructConverter::convertAvroValue( const avro::GenericDatum & datum, const FieldEntry & entry, StructPtr & struct_ )
{
    // datum.type() automatically handles unions - no need to unwrap
    if( datum.type() == avro::AVRO_NULL )
        return;

    const char * fieldname = entry.avroFieldName.c_str();
    auto fieldType = entry.sField -> type();

    switch( fieldType -> type() )
    {
        case CspType::Type::BOOL:
            entry.sField -> setValue( struct_.get(), extractValue<bool>( datum, fieldname ) );
            break;
        case CspType::Type::INT32:
            entry.sField -> setValue( struct_.get(), extractValue<int32_t>( datum, fieldname ) );
            break;
        case CspType::Type::INT64:
            entry.sField -> setValue( struct_.get(), extractValue<int64_t>( datum, fieldname ) );
            break;
        case CspType::Type::DOUBLE:
            entry.sField -> setValue( struct_.get(), extractValue<double>( datum, fieldname ) );
            break;
        case CspType::Type::STRING:
            entry.sField -> setValue( struct_.get(), extractValue<std::string>( datum, fieldname ) );
            break;
        case CspType::Type::DATE:
            entry.sField -> setValue( struct_.get(), extractValue<Date>( datum, fieldname ) );
            break;
        case CspType::Type::DATETIME:
            entry.sField -> setValue( struct_.get(), extractValue<DateTime>( datum, fieldname ) );
            break;
        case CspType::Type::ENUM:
        {
            if( datum.type() != avro::AVRO_ENUM )
                CSP_THROW( TypeError, "expected ENUM type for avro field " << fieldname );
            const avro::GenericEnum & e = datum.value<avro::GenericEnum>();
            auto & cspEnumType = static_cast<const CspEnumType &>( *fieldType );
            entry.sField -> setValue( struct_.get(), cspEnumType.meta() -> fromString( e.symbol().c_str() ) );
            break;
        }
        case CspType::Type::STRUCT:
        {
            if( datum.type() != avro::AVRO_RECORD )
                CSP_THROW( TypeError, "expected RECORD type for avro field " << fieldname );
            
            const avro::GenericRecord & record = datum.value<avro::GenericRecord>();
            const CspStructType & sType = static_cast<const CspStructType &>( *fieldType );
            auto nestedStruct = sType.meta() -> create();

            if( entry.nestedFields )
            {
                for( auto & [nestedAvroName, nestedEntry] : *entry.nestedFields )
                {
                    if( record.hasField( nestedAvroName ) )
                    {
                        convertAvroValue( record.field( nestedAvroName ), nestedEntry, nestedStruct );
                    }
                }
            }
            entry.sField -> setValue( struct_.get(), nestedStruct );
            break;
        }
        case CspType::Type::ARRAY:
        {
            const CspArrayType & arrType = static_cast<const CspArrayType &>( *fieldType );
            const CspType & elemType = *arrType.elemType();

            SupportedArrayCspTypeSwitch::invoke(
                &elemType, [this, &datum, &entry, &struct_, fieldname, &elemType]( auto tag )
                {
                    using T = typename decltype(tag)::type;
                    entry.sField -> setValue( struct_.get(), extractArray<T>( datum, fieldname, elemType ) );
                }
            );
            break;
        }
        default:
            CSP_THROW( NotImplemented, "Unsupported type for avro field " << fieldname );
    }
}

AvroMessageStructConverter::AvroMessageStructConverter( const CspTypePtr & type,
                                                        const Dictionary & properties ) : MessageStructConverter( type, properties )
{
    if( type -> type() != CspType::Type::STRUCT )
        CSP_THROW( TypeError, "AvroMessageStructConverter expects type struct got " << type -> type() );

    std::string schemaStr = properties.get<std::string>( "avro_schema" );
    std::istringstream iss( schemaStr );
    avro::compileJsonSchema( iss, m_schema );

    const Dictionary & fieldMap = *properties.get<DictionaryPtr>( "field_map" );
    m_datetimeType = DateTimeWireType( properties.get<std::string>( "datetime_type" ) );
    m_fields = buildFields( static_cast<const CspStructType &>( *type ), fieldMap );
}

AvroMessageStructConverter::Fields AvroMessageStructConverter::buildFields( const CspStructType & type, const Dictionary & fieldMap )
{
    AvroMessageStructConverter::Fields out;

    for( auto it = fieldMap.begin(); it != fieldMap.end(); ++it )
    {
        auto & avroFieldName = it.key();

        std::string structField;
        DictionaryPtr nestedFieldMap;

        if( it.hasValue<std::string>() )
            structField = it.value<std::string>();
        else
        {
            if( !it.hasValue<DictionaryPtr>() )
                CSP_THROW( TypeError, "fieldMap expected string or dict for field " << avroFieldName << " on struct " << type.meta() -> name() );
            auto nestedDict = it.value<DictionaryPtr>();
            if( nestedDict -> size() != 1 )
                CSP_THROW( ValueError, "Expected nested fieldmap for incoming Avro field " << avroFieldName << " to have a single key : map entry" );
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

        out.emplace( avroFieldName, FieldEntry{ sField, avroFieldName, nestedFields } );
    }
    return out;
}

csp::StructPtr AvroMessageStructConverter::asStruct( void * bytes, size_t size )
{
    auto inputStream = avro::memoryInputStream( static_cast<const uint8_t*>( bytes ), size );
    avro::DecoderPtr decoder = avro::binaryDecoder();
    decoder -> init( *inputStream );

    avro::GenericDatum datum( m_schema );
    avro::GenericReader::read( *decoder, datum, m_schema );

    if( datum.type() != avro::AVRO_RECORD )
        CSP_THROW( ValueError, "Expected Avro record at top level" );

    const avro::GenericRecord & record = datum.value<avro::GenericRecord>();
    StructPtr data = m_structMeta -> create();

    for( auto & [avroFieldName, entry] : m_fields )
    {
        if( record.hasField( avroFieldName ) )
        {
            convertAvroValue( record.field( avroFieldName ), entry, data );
        }
    }

    return data;
}

}
