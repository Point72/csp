#ifndef _IN_CSP_ADAPTERS_UTILS_JSONMESSAGESTRUCTCONVERTER_H
#define _IN_CSP_ADAPTERS_UTILS_JSONMESSAGESTRUCTCONVERTER_H

#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/core/Hash.h>
#include <csp/engine/Dictionary.h>
#include <rapidjson/document.h>
#include <list>
#include <string>
#include <unordered_map>

namespace csp::adapters::utils
{

class JSONMessageStructConverter: public MessageStructConverter
{
public:
    JSONMessageStructConverter( const CspTypePtr & type, const Dictionary & properties );

    csp::StructPtr asStruct( void * bytes, size_t size ) final;

    static MessageStructConverter * create( const CspTypePtr & type, const Dictionary & properties )
    {
        return new JSONMessageStructConverter( type, properties );
    }

private:

    //map of json field -> struct field ptr
    //we keep a hash since rapidjson Document field lookups are O(n)!
    struct FieldEntry
    {
        StructFieldPtr sField;
        std::shared_ptr<std::unordered_map<const char*,FieldEntry,csp::hash::CStrHash,csp::hash::CStrEq>> nestedFields;
    };

    using Fields = std::unordered_map<const char*,FieldEntry,csp::hash::CStrHash,csp::hash::CStrEq>;

    Fields buildFields( const CspStructType & type, const Dictionary & fieldMap );

    //T* only used for vector overloads
    template<typename T>
    T convertJSON( const char * fieldname, const rapidjson::Value & v, T * );

    template<typename T>
    T convertJSON( const char * fieldname, const CspType & type, const FieldEntry & entry, const rapidjson::Value & v, T * foo )
    {
        return convertJSON( fieldname, v, foo );
    }

    template<typename StorageT>
    std::vector<StorageT> convertJSON( const char * fieldname, const CspType & type, const FieldEntry & entry, const rapidjson::Value & v, std::vector<StorageT> * );

    
    Fields           m_fields;
    DateTimeWireType m_datetimeType;
    std::list<std::string> m_jsonkeys; //intentionally stored as list so they dont invalidate on push
};

}

#endif //_IN_CSP_ADAPTERS_ACTIVEMQ_JSONMESSAGESTRUCTCONVERTER_H
