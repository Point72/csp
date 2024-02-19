#include <csp/adapters/websocket_client/WSClientInputAdapter.h>

namespace csp::adapters::wsclient
{

WSClientInputAdapter::WSClientInputAdapter(
    Engine * engine,
    CspTypePtr & type,
    PushMode pushMode,
    const Dictionary & properties
) : PushInputAdapter(engine, type, pushMode)
{
    // TODO: should I support bytes?
    if( type -> type() != CspType::Type::STRUCT &&
        type -> type() != CspType::Type::STRING )
        CSP_THROW( RuntimeException, "Unsupported type: " << type -> type() );
    
    if( properties.exists( "meta_field_map" ) )
    {
        // const CspStructType & structType = static_cast<const CspStructType &>( *type );
        const Dictionary & metaFieldMap = *properties.get<DictionaryPtr>( "meta_field_map" );

        if( !metaFieldMap.empty() && type -> type() != CspType::Type::STRUCT )
            CSP_THROW( ValueError, "meta_field_map is not supported on non-struct types" );
    }

    m_converter = utils::MessageStructConverterCache::instance().create( type, properties );
};

void WSClientInputAdapter::processMessage( message_ptr message, csp::PushBatch* batch ) 
{

    if( type() -> type() == CspType::Type::STRUCT )
    {
        auto payload = message -> get_payload();
        auto tick = m_converter -> asStruct( &payload, payload.length() );
        pushTick( std::move(tick), batch );
    } else if ( type() -> type() == CspType::Type::STRING )
    {
        pushTick( std::move(message->get_payload()), batch );
    }

}

} // namespace csp::adapters::wsclient
