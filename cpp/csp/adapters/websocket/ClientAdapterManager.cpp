#include <csp/adapters/websocket/ClientAdapterManager.h>

namespace csp {

INIT_CSP_ENUM( adapters::websocket::ClientStatusType,
               "ACTIVE",
               "GENERIC_ERROR",
               "CONNECTION_FAILED",
               "CLOSED",
               "MESSAGE_SEND_FAIL",
);

}

// With TLS
namespace csp::adapters::websocket {

ClientAdapterManager::ClientAdapterManager( Engine* engine, const Dictionary & properties ) 
: AdapterManager( engine ), 
    m_active( false ), 
    m_shouldRun( false ), 
    m_endpoint( std::make_unique<WebsocketEndpoint>( properties ) ),
    m_inputAdapter( nullptr ), 
    m_outputAdapter( nullptr ),
    m_updateAdapter( nullptr ),
    m_thread( nullptr ), 
    m_properties( properties ) 
{ };

ClientAdapterManager::~ClientAdapterManager()
{ };

void ClientAdapterManager::start( DateTime starttime, DateTime endtime )
{
    AdapterManager::start( starttime, endtime );

    m_shouldRun = true;
    m_endpoint -> setOnOpen(
        [ this ]() {
            m_active = true;
            pushStatus( StatusLevel::INFO, ClientStatusType::ACTIVE, "Connected successfully" );
        }
    );
    m_endpoint -> setOnFail(
        [ this ]( const std::string& reason ) {
            std::stringstream ss;
            ss << "Connection Failure: " << reason;
            m_active = false;
            pushStatus( StatusLevel::ERROR, ClientStatusType::CONNECTION_FAILED, ss.str() );
        } 
    );
    if( m_inputAdapter ) {
        m_endpoint -> setOnMessage(
            [ this ]( void* c, size_t t ) {
                PushBatch batch( m_engine -> rootEngine() );
                m_inputAdapter -> processMessage( c, t, &batch );
            }
        );
    } else {
        // if a user doesn't call WebsocketAdapterManager.subscribe, no inputadapter will be created
        // but we still need something to avoid on_message_cb not being set in the endpoint.
        m_endpoint -> setOnMessage( []( void* c, size_t t ){} );
    }
    m_endpoint -> setOnClose(
        [ this ]() {
            m_active = false;
            pushStatus( StatusLevel::INFO, ClientStatusType::CLOSED, "Connection closed" );
        }
    );
    m_endpoint -> setOnSendFail(
        [ this ]( const std::string& s ) {
            std::stringstream ss;
            ss << "Failed to send: " << s;
            pushStatus( StatusLevel::ERROR, ClientStatusType::MESSAGE_SEND_FAIL, ss.str() );
        }
    );

    m_thread = std::make_unique<std::thread>( [ this ]() { 
        while( m_shouldRun )
        {
            m_endpoint -> run();
            m_active = false;
            if( m_shouldRun ) sleep( m_properties.get<TimeDelta>( "reconnect_interval" ) );
        }
    });
};

void ClientAdapterManager::stop() {
    AdapterManager::stop();

    m_shouldRun=false; 
    if( m_active ) m_endpoint->stop();
    if( m_thread ) m_thread->join();
};

PushInputAdapter* ClientAdapterManager::getInputAdapter(CspTypePtr & type, PushMode pushMode, const Dictionary & properties)
{
    if (m_inputAdapter == nullptr)
    {
        m_inputAdapter = m_engine -> createOwnedObject<ClientInputAdapter>(
            // m_engine,
            type,
            pushMode,
            properties    
        );
    }
    return m_inputAdapter;
};

OutputAdapter* ClientAdapterManager::getOutputAdapter()
{
    if (m_outputAdapter == nullptr) m_outputAdapter = m_engine -> createOwnedObject<ClientOutputAdapter>(*m_endpoint);

    return m_outputAdapter;
}

OutputAdapter * ClientAdapterManager::getHeaderUpdateAdapter()
{
    if (m_updateAdapter == nullptr) m_updateAdapter = m_engine -> createOwnedObject<ClientHeaderUpdateOutputAdapter>( m_endpoint -> getProperties() );

    return m_updateAdapter;
}

DateTime ClientAdapterManager::processNextSimTimeSlice( DateTime time )
{
    // no sim data
    return DateTime::NONE();
}

}
