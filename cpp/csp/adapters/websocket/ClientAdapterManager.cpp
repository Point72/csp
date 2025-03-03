#include <csp/adapters/websocket/ClientAdapterManager.h>

namespace csp::adapters::websocket {

ClientAdapterManager::ClientAdapterManager( Engine* engine, const Dictionary & properties ) 
: AdapterManager( engine ), 
  m_properties( properties )
{ }

ClientAdapterManager::~ClientAdapterManager()
{ }

WebsocketEndpointManager* ClientAdapterManager::getWebsocketManager(){
    if( m_endpointManager == nullptr )
        return nullptr;
    return m_endpointManager.get();
}

void ClientAdapterManager::start(DateTime starttime, DateTime endtime) {
    AdapterManager::start(starttime, endtime);
    if (m_endpointManager != nullptr)
        m_endpointManager -> start(starttime, endtime);
}

void ClientAdapterManager::stop() {
    AdapterManager::stop();
    if (m_endpointManager != nullptr)
        m_endpointManager -> stop();
}

PushInputAdapter* ClientAdapterManager::getInputAdapter(CspTypePtr & type, PushMode pushMode, const Dictionary & properties)
{   
    if (m_endpointManager == nullptr)
        m_endpointManager = std::make_unique<WebsocketEndpointManager>(this, m_properties, m_engine);
    return m_endpointManager -> getInputAdapter( type, pushMode, properties );
}

OutputAdapter* ClientAdapterManager::getOutputAdapter( const Dictionary & properties )
{
    if (m_endpointManager == nullptr)
        m_endpointManager = std::make_unique<WebsocketEndpointManager>(this, m_properties, m_engine);
    return m_endpointManager -> getOutputAdapter( properties );
}

OutputAdapter * ClientAdapterManager::getHeaderUpdateAdapter()
{
   if (m_endpointManager == nullptr)
        m_endpointManager = std::make_unique<WebsocketEndpointManager>(this, m_properties, m_engine);
    return m_endpointManager -> getHeaderUpdateAdapter();
}

OutputAdapter * ClientAdapterManager::getConnectionRequestAdapter( const Dictionary & properties )
{
    if (m_endpointManager == nullptr)
        m_endpointManager = std::make_unique<WebsocketEndpointManager>(this, m_properties, m_engine);
    return m_endpointManager -> getConnectionRequestAdapter( properties );
}

DateTime ClientAdapterManager::processNextSimTimeSlice( DateTime time )
{
    // no sim data
    return DateTime::NONE();
}

}
