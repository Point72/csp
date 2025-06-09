#ifndef _IN_CSP_ADAPTERS_UTILS_JSONMESSAGEWRITER_H
#define _IN_CSP_ADAPTERS_UTILS_JSONMESSAGEWRITER_H

#include <csp/adapters/utils/MessageWriter.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PartialSwitchCspType.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <string>

namespace csp::adapters::utils
{

class JSONMessageWriter : public MessageWriter
{
public:
    using SupportedCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL,
                                                        csp::CspType::Type::UINT8,
                                                        csp::CspType::Type::INT16,
                                                        csp::CspType::Type::INT32,
                                                        csp::CspType::Type::INT64,
                                                        csp::CspType::Type::DOUBLE,
                                                        csp::CspType::Type::DATE,
                                                        csp::CspType::Type::DATETIME,
                                                        csp::CspType::Type::ENUM,
                                                        csp::CspType::Type::STRING,
                                                        csp::CspType::Type::STRUCT,
                                                        csp::CspType::Type::ARRAY
                                                        >;

    using SupportedArrayCspTypeSwitch = PartialSwitchCspType<csp::CspType::Type::BOOL,
            csp::CspType::Type::UINT8,
            csp::CspType::Type::INT16,
            csp::CspType::Type::INT32,
            csp::CspType::Type::INT64,
            csp::CspType::Type::DOUBLE,
            csp::CspType::Type::DATETIME,
            csp::CspType::Type::ENUM,
            csp::CspType::Type::STRING
    >;

    JSONMessageWriter( const Dictionary & properties )
    {
        m_doc.SetObject();
        m_datetimeWireType = utils::DateTimeWireType( properties.get<std::string>( "datetime_type" ) );
    }

    template<typename T>
    void setField( const std::string & field, const T & value, const CspType & type, const FieldEntry & entry );

    std::pair<const void *,size_t> finalize() override
    {
        using Writer = rapidjson::Writer<rapidjson::StringBuffer,rapidjson::UTF8<>,rapidjson::UTF8<>,
                                         rapidjson::CrtAllocator,rapidjson::kWriteNanAndInfFlag>;
        m_stringBuffer.Clear();
        Writer writer( m_stringBuffer );
        m_doc.Accept( writer );
        //reset document
        //Note we have to explicitly clear the memory pool to avoid leaking!
        m_doc.GetAllocator().Clear();
        m_doc.SetObject();

        return {m_stringBuffer.GetString(),m_stringBuffer.GetSize()};
    }

private:
    void processTickImpl( const OutputDataMapper & dataMapper, const TimeSeriesProvider * sourcets ) override
    {
        dataMapper.apply( *this, sourcets );
    }

    template<typename T>
    inline auto convertValue( const T & value )
    { 
        return value;
    }

    template<typename T>
    inline auto convertValue( const T & value, const CspType & type, const FieldEntry & entry )
    {
        return convertValue( value );
    }


    template<typename StorageT>
    auto convertValue( const std::vector<StorageT> & value, const CspType & type, const FieldEntry & entry );

    rapidjson::Document     m_doc;
    rapidjson::StringBuffer m_stringBuffer;
    utils::DateTimeWireType m_datetimeWireType;
};

template<>
inline auto JSONMessageWriter::convertValue( const std::string & value )
{
    return rapidjson::StringRef( value.c_str() ); 
}

template<>
inline auto JSONMessageWriter::convertValue( const csp::Date & value )
{
    return rapidjson::Value( value.asYYYYMMDD().c_str(), m_doc.GetAllocator() ); 
}

template<>
inline auto JSONMessageWriter::convertValue( const csp::DateTime & value )
{
    switch( m_datetimeWireType )
    {
        case utils::DateTimeWireType::UINT64_NANOS:
            return ( uint64_t ) value.asNanoseconds();
        case utils::DateTimeWireType::UINT64_MICROS:
            return ( uint64_t ) value.asMicroseconds();
        case utils::DateTimeWireType::UINT64_MILLIS:
            return ( uint64_t ) value.asMilliseconds();
        case utils::DateTimeWireType::UINT64_SECONDS:
            return ( uint64_t ) value.asSeconds();

        default:
            CSP_THROW( NotImplemented, "datetime wire type " << m_datetimeWireType << " not supported for json msg publishing" );
    }
}

template<>
inline auto JSONMessageWriter::convertValue( const csp::TimeDelta & value )
{
    return rapidjson::Value( value.asNanoseconds() ); 
}

template<>
inline auto JSONMessageWriter::convertValue( const csp::CspEnum & value, const CspType & type, const FieldEntry & entry )
{
    return rapidjson::StringRef( value.name().c_str() );
}

template<typename StorageT>
inline auto JSONMessageWriter::convertValue( const std::vector<StorageT> & value, const CspType & type, const FieldEntry & entry )
{
    auto & allocator = m_doc.GetAllocator();
    rapidjson::Value array( rapidjson::kArrayType );
    size_t sz = value.size();

    const CspType & elemType = *static_cast<const CspArrayType &>( type ).elemType();

    using ElemT = typename CspType::Type::toCArrayElemType<StorageT>::type;

    //iterating by index for vector<bool> support
    for( size_t index = 0; index < sz; ++index )
    {
        //Note this passes an empty FieldEntry / wont work on vector of structs
        array.PushBack( convertValue<ElemT>( value[index], elemType, {} ), allocator );
    }
    return array;
}

template<>
inline auto JSONMessageWriter::convertValue( const StructPtr & struct_, const CspType & type, const FieldEntry & entry )
{
    rapidjson::Value jValue( rapidjson::kObjectType );
    for( auto & nestedEntry : *entry.nestedFields )
    {
        if( !nestedEntry.sField -> isSet( struct_.get() ) )
            continue;

        
        SupportedCspTypeSwitch::template invoke<SupportedArrayCspTypeSwitch>(
            nestedEntry.sField -> type().get(),
            [ & ]( auto tag )
            {
                using T = typename decltype(tag)::type;
                jValue.AddMember( rapidjson::StringRef( nestedEntry.outField.c_str() ), convertValue( nestedEntry.sField -> value<T>( struct_.get() ), *nestedEntry.sField -> type(), nestedEntry ), m_doc.GetAllocator() );
            } );
    };
    return jValue;
}


template<typename T>
inline void JSONMessageWriter::setField( const std::string & field, const T & value, const CspType & type, const FieldEntry & entry )
{
    m_doc.AddMember( rapidjson::StringRef( field.c_str() ), convertValue( value, type, entry ), m_doc.GetAllocator() );
}

}

#endif
