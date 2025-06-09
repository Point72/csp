#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/adapters/utils/JSONMessageStructConverter.h>
#include <csp/adapters/utils/RawBytesMessageStructConverter.h>

namespace csp::adapters::utils
{

MessageStructConverter::MessageStructConverter( const CspTypePtr & type, const Dictionary & properties ) : m_type( type )
{
    if( type -> type() == CspType::Type::STRUCT )
        m_structMeta = std::static_pointer_cast<const CspStructType>( type ) -> meta();
}

MessageStructConverterCache::MessageStructConverterCache()
{
    registerConverter( "RAW_BYTES", &RawBytesMessageStructConverter::create );
    registerConverter( "JSON",      &JSONMessageStructConverter::create );
}

bool MessageStructConverterCache::registerConverter( std::string protocol, Creator creator )
{
    if( m_creators.find( protocol ) != m_creators.end() )
        CSP_THROW( RuntimeException, "Attempted to register creator for MessageStructConverter type " << protocol << " more than once" );

    m_creators[ protocol ] = creator;
    return true;
}

bool MessageStructConverterCache::hasConverter( std::string protocol ) const
{
    return m_creators.find( protocol ) != m_creators.end();
}

MessageStructConverterCache & MessageStructConverterCache::instance()
{
    static MessageStructConverterCache s_instance;
    return s_instance;
}

MessageStructConverterPtr MessageStructConverterCache::create( const CspTypePtr & type, const Dictionary & properties )
{
    std::lock_guard<std::mutex> guard( m_cacheMutex );

    auto rv = m_cache.emplace( CacheKey{ type.get(), properties }, nullptr );
    if( !rv.second )
        return rv.first -> second;

    auto protocol = properties.get<std::string>( "protocol" );
    auto creatorIt = m_creators.find( protocol );
    if( creatorIt == m_creators.end() )
        CSP_THROW( ValueError, "MessageStructConverter for type " << protocol << " is not defined" );

    auto result = std::shared_ptr<MessageStructConverter>( creatorIt -> second( type, properties ) );
    rv.first -> second = result;
    return rv.first -> second;
}

}
