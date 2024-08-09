#include <csp/adapters/websocket/ClientInputAdapter.h>

namespace csp::adapters::websocket
{

ClientInputAdapter::ClientInputAdapter(
    Engine * engine,
    CspTypePtr & type,
    PushMode pushMode,
    const Dictionary & properties
) : PushInputAdapter(engine, type, pushMode)
{
    if( type -> type() != CspType::Type::STRUCT &&
        type -> type() != CspType::Type::STRING )
        CSP_THROW( RuntimeException, "Unsupported type: " << type -> type() );

    if( properties.exists( "meta_field_map" ) )
    {
        const Dictionary & metaFieldMap = *properties.get<DictionaryPtr>( "meta_field_map" );

        if( !metaFieldMap.empty() && type -> type() != CspType::Type::STRUCT )
            CSP_THROW( ValueError, "meta_field_map is not supported on non-struct types" );
    }

    m_converter = adapters::utils::MessageStructConverterCache::instance().create( type, properties );
};

void ClientInputAdapter::processMessage( void* c, size_t t, PushBatch* batch ) 
{

    if( dataType() -> type() == CspType::Type::STRUCT )
    {
        auto tick = m_converter -> asStruct( c, t );
        pushTick( std::move(tick), batch );
    } else if ( dataType() -> type() == CspType::Type::STRING )
    {
        pushTick( std::string((char const*)c, t), batch );
    }

}

}