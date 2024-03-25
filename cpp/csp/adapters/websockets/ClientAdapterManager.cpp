#include <csp/adapters/websockets/ClientAdapterManager.h>

#include <csp/core/Platform.h>
#include <chrono>
#include <iomanip>
#include <iostream>


using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

namespace csp {

INIT_CSP_ENUM( adapters::websockets::ClientStatusType,
               "ACTIVE",
               "GENERIC_ERROR",
               "CONNECTION_FAILED",
               "CLOSED",
);

}

// With TLS
namespace csp::adapters::websockets {

ClientAdapterManager::ClientAdapterManager( Engine* engine, const Dictionary & properties 
) : AdapterManager( engine ), 
    m_active(false), 
    m_shouldRun(false), 
    m_endpoint(nullptr),
    m_inputAdapter(nullptr), 
    m_outputAdapter(nullptr),
    m_updateAdapter(nullptr),
    m_thread(nullptr), 
    m_properties(properties) 
{
    if (m_properties.get<bool>("use_tls")) {
        m_endpoint = new WebsocketEndpointTLS(properties);
    } else {
        m_endpoint = new WebsocketEndpointNoTLS(properties);
    }

    m_endpoint->setOnMessageCb(std::move([this](std::string msg) {
        if (m_inputAdapter != nullptr)
        {
            PushBatch batch( m_engine -> rootEngine() );
            m_inputAdapter->processMessage(msg, &batch);
        }
    }));
    m_endpoint->setOnOpenCb(std::move([this](){
        m_active = true;
        pushStatus(StatusLevel::INFO, ClientStatusType::ACTIVE, "Connected successfully");
    }));
    m_endpoint->setOnFailCb(std::move([this](){
        m_active = false;
        pushStatus(StatusLevel::ERROR, ClientStatusType::CONNECTION_FAILED, "Connection failed, will try to reconnect");
    }));
    m_endpoint->setOnCloseCb(std::move([this](){
        m_active = false;
        pushStatus(StatusLevel::INFO, ClientStatusType::CLOSED, "Connection closed");
    }));

};

ClientAdapterManager::~ClientAdapterManager()
{ };

void ClientAdapterManager::start( DateTime starttime, DateTime endtime )
{
    AdapterManager::start( starttime, endtime );
    // start the bg thread
    m_shouldRun = true;
    m_thread = std::make_unique<std::thread>( [ this ](){ 
        while (m_shouldRun)
        {
            m_endpoint->run();
            m_active=false;
            if(m_shouldRun) std::this_thread::sleep_for( std::chrono::seconds(m_properties.get<TimeDelta>("reconnect_interval").asSeconds()) );
        }
    });
};

void ClientAdapterManager::stop() {
    AdapterManager::stop();

    m_shouldRun=false; 
    if(m_active) {
        m_endpoint->close();
    }

    if(m_thread) {
        m_thread->join();
    }
};

PushInputAdapter* ClientAdapterManager::getInputAdapter(CspTypePtr & type, PushMode pushMode, const Dictionary & properties)
{
    if (m_inputAdapter == nullptr)
    {
        m_inputAdapter = m_engine->createOwnedObject<ClientInputAdapter>(
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
    if (m_outputAdapter == nullptr)
    {
        m_outputAdapter = m_engine->createOwnedObject<ClientOutputAdapter>(m_endpoint);
    }

    return m_outputAdapter;
}

OutputAdapter * ClientAdapterManager::getHeaderUpdateAdapter()
{
    if (m_updateAdapter == nullptr)
    {
        m_updateAdapter = m_engine->createOwnedObject<ClientHeaderUpdateAdapter>(
            m_endpoint->getProperties()
        );
    }

    return m_updateAdapter;
}

DateTime ClientAdapterManager::processNextSimTimeSlice( DateTime time )
{
    // no sim data
    return DateTime::NONE();
}

}
