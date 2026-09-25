#ifndef _IN_CSP_ADAPTERS_UTILS_AVROMESSAGESTRUCTCONVERTER_H
#define _IN_CSP_ADAPTERS_UTILS_AVROMESSAGESTRUCTCONVERTER_H

#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/adapters/utils/AvroIncludes.h>
#include <csp/core/Hash.h>
#include <csp/engine/Dictionary.h>

#include <list>
#include <string>
#include <unordered_map>

namespace csp::adapters::utils
{

class AvroMessageStructConverter: public MessageStructConverter
{
public:
    AvroMessageStructConverter( const CspTypePtr & type, const Dictionary & properties );

    csp::StructPtr asStruct( void * bytes, size_t size ) final;

    static MessageStructConverter * create( const CspTypePtr & type, const Dictionary & properties )
    {
        return new AvroMessageStructConverter( type, properties );
    }

private:
    struct FieldEntry
    {
        StructFieldPtr sField;
        std::string avroFieldName;
        std::shared_ptr<std::unordered_map<std::string, FieldEntry>> nestedFields;
    };

    using Fields = std::unordered_map<std::string, FieldEntry>;

    Fields buildFields( const CspStructType & type, const Dictionary & fieldMap );

    void convertAvroValue( const avro::GenericDatum & datum, const FieldEntry & entry, StructPtr & struct_ );

    template<typename T>
    T extractValue( const avro::GenericDatum & datum, const char * fieldname );

    template<typename StorageT>
    std::vector<StorageT> extractArray( const avro::GenericDatum & datum, const char * fieldname, const CspType & elemType );

    Fields             m_fields;
    DateTimeWireType   m_datetimeType;
    avro::ValidSchema  m_schema;
};

}

#endif
