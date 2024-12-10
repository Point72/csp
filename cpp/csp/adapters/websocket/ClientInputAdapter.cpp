#include <csp/adapters/websocket/ClientInputAdapter.h>
namespace csp::adapters::websocket
{

ClientInputAdapter::ClientInputAdapter(
    Engine * engine,
    CspTypePtr & type,
    PushMode pushMode,
    const Dictionary & properties,
    bool dynamic
) : PushInputAdapter(engine, type, pushMode),
    m_dynamic( dynamic ) 
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
    if ( m_dynamic ){
        auto& actual_type = static_cast<const CspStructType &>( *type );
        auto& nested_type = actual_type.meta()-> field( "msg" ) -> type();

        m_converter = adapters::utils::MessageStructConverterCache::instance().create( nested_type, properties );
    }
    else
        m_converter = adapters::utils::MessageStructConverterCache::instance().create( type, properties );
};

void ClientInputAdapter::processMessage( const std::string& source, void * c, size_t t, PushBatch* batch ) 
{
    if ( m_dynamic ){
        auto& actual_type = static_cast<const CspStructType &>( *dataType() );
        auto& nested_type = actual_type.meta()-> field( "msg" ) -> type();
        auto true_val = actual_type.meta() -> create();
        actual_type.meta()->field("uri")->setValue( true_val.get(), source );

        if( nested_type -> type() == CspType::Type::STRUCT )
        {
            auto tick = m_converter -> asStruct( c, t );
            actual_type.meta()->field("msg")->setValue( true_val.get(), std::move(tick) );

            pushTick( std::move(true_val), batch );
        } else if ( nested_type -> type() == CspType::Type::STRING )
        {
            auto msg =  std::string((char const*)c, t);
            actual_type.meta()->field("msg")->setValue( true_val.get(), msg );

            pushTick( std::move(true_val), batch );
        }

    }
    else{
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

}