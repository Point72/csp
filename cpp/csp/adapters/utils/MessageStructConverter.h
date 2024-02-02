#ifndef _IN_CSP_ADAPTERS_UTILS_MESSAGESTRUCTCONVERTER_H
#define _IN_CSP_ADAPTERS_UTILS_MESSAGESTRUCTCONVERTER_H

#include <csp/adapters/utils/MessageEnums.h>
#include <csp/core/Enum.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Struct.h>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

namespace csp::adapters::utils
{

class MessageStructConverter
{
public:
    MessageStructConverter( const CspTypePtr & type, const Dictionary & properties );
    virtual csp::StructPtr asStruct( void * bytes, size_t size ) = 0;

    virtual MsgProtocol protocol() const = 0;

    StructMetaPtr structMeta() { return m_structMeta; }

protected:
    CspTypePtr    m_type;
    StructMetaPtr m_structMeta;

private:
    using FieldEntry = std::pair<std::string,StructFieldPtr>;
    using Fields     = std::vector<FieldEntry>;

    Fields m_propertyFields;
};

using MessageStructConverterPtr=std::shared_ptr<MessageStructConverter>;

//This ensures we dont recreate converters unnecessarily for say subscription by symbol with the same
//conversion onformation
class MessageStructConverterCache
{
public:
    MessageStructConverterCache();

    static MessageStructConverterCache & instance();

    MessageStructConverterPtr create( const CspTypePtr &, const Dictionary & properties );

    using Creator = std::function<MessageStructConverter*( const CspTypePtr &, const Dictionary & )>;

    bool registerConverter( MsgProtocol protocol, Creator creator );

private:
    using CacheKey = std::pair<const CspType*,Dictionary>;
    using Cache = std::unordered_map<CacheKey,MessageStructConverterPtr,csp::hash::hash_pair>;

    std::mutex m_cacheMutex;
    Cache      m_cache;
    Creator    m_creators[ MsgProtocol::NUM_TYPES ];
};

}

#endif
