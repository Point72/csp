#include <csp/adapters/websocket/ClientHeaderUpdateAdapter.h>

namespace csp::adapters::websocket {

ClientHeaderUpdateOutputAdapter::ClientHeaderUpdateOutputAdapter(
    Engine * engine,
    Dictionary& properties
) : OutputAdapter( engine ), m_properties( properties )
{ };

void ClientHeaderUpdateOutputAdapter::executeImpl()
{
    DictionaryPtr headers = m_properties.get<DictionaryPtr>("headers");
    for( auto& update : input() -> lastValueTyped<std::vector<WebsocketHeaderUpdate::Ptr>>() )
    { 
        if( update -> key_isSet() && update -> value_isSet() ) headers->update( update->key(), update->value() ); 
    }
};

}