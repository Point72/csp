#include <csp/adapters/websocket_client/WSClientAdapterManager.h>

#include <csp/core/Platform.h>
#include <iostream>

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

namespace csp {

INIT_CSP_ENUM( csp::adapters::wsclient::WSClientStatusType,
               "ACTIVE",
               "GENERIC_ERROR",
               "CONNECTION_FAILED"
);

}



namespace csp::adapters::wsclient {

WSClientAdapterManager::WSClientAdapterManager( csp::Engine* engine, const Dictionary & properties 
) : csp::AdapterManager( engine ), m_inputAdapter(nullptr), m_thread(nullptr), m_threadActive(false), m_active(false), m_shouldRun(false), m_properties(properties), m_outputAdapter(nullptr)
{
    if (m_properties.get<bool>("verbose_log")) {
        m_client.set_access_channels(websocketpp::log::alevel::all);
        // m_client.clear_access_channels(websocketpp::log::alevel::frame_payload);
    } else {
        m_client.clear_access_channels(websocketpp::log::alevel::all);
    }
    m_client.init_asio();

    m_client.set_open_handler(bind(&WSClientAdapterManager::onOpen, this, ::_1));
    m_client.set_message_handler(bind(&WSClientAdapterManager::onMessage, this, ::_1, ::_2));
    m_client.set_fail_handler(bind(&WSClientAdapterManager::onFail, this, ::_1));

};

WSClientAdapterManager::~WSClientAdapterManager()
{
    // I should probably check if the thread is active...
    // m_thread->join();
    if (m_threadActive) {
        m_thread->join();
    }
};

void WSClientAdapterManager::onMessage( websocketpp::connection_hdl hdl, message_ptr msg )
{
    if (m_inputAdapter != nullptr)
    {
        csp::PushBatch batch( m_engine -> rootEngine() );
        m_inputAdapter->processMessage(msg, &batch);
    }
};

void WSClientAdapterManager::onOpen( websocketpp::connection_hdl hdl )
{
    m_active = true;
    pushStatus(StatusLevel::INFO, WSClientStatusType::ACTIVE, "Connected successfully");
}

void WSClientAdapterManager::onFail( websocketpp::connection_hdl hdl )
{
    m_active = false;
    pushStatus(StatusLevel::ERROR, WSClientStatusType::GENERIC_ERROR, "Could not connect");
};

void WSClientAdapterManager::start( DateTime starttime, DateTime endtime )
{
    AdapterManager::start( starttime, endtime );
    // start the bg thread
    m_threadActive = true;
    m_shouldRun = true;
    m_thread = std::make_unique<std::thread>( [ this ](){ 
        while (m_shouldRun)
        {
            this->innerLoop();
            m_active=false;
            pushStatus(StatusLevel::ERROR, WSClientStatusType::GENERIC_ERROR, "Reconnecting...");
            if(m_shouldRun) std::this_thread::sleep_for( std::chrono::seconds(m_properties.get<binding_int_t>("reconnect_seconds")) );
        }
    });
};

void WSClientAdapterManager::innerLoop()
{
    
    auto uri = m_properties.get<std::string>("uri");
    websocketpp::lib::error_code ec;
    client::connection_ptr con = m_client.get_connection(uri, ec);
    const Dictionary &headers = *m_properties.get<DictionaryPtr>("headers");
    for( auto it = headers.begin(); it != headers.end(); ++it )
    {
        const std::string key = it.key();
        const std::string value = headers.get<std::string>( key );
        con.get()->append_header(key, value);
    }
    if (ec) {
        CSP_THROW(RuntimeException, "could not create connection because: " << ec.message());
    }

    m_client.connect(con);
    m_hdl = con->get_handle();
    m_client.run();
    m_client.reset();
}

void WSClientAdapterManager::stop() {
    AdapterManager::stop();
    m_shouldRun=false; 
    if(m_active) {
        websocketpp::lib::error_code ec;
        m_client.close(m_hdl, websocketpp::close::status::going_away, "Good bye", ec);
        if (ec) {
            CSP_THROW(RuntimeException, "could not close connection because: " << ec.message());
        }
    }

    if(m_threadActive) {
        m_thread->join();
        m_threadActive=false;
    }
};

PushInputAdapter* WSClientAdapterManager::getInputAdapter(CspTypePtr & type, PushMode pushMode, const Dictionary & properties)
{
    if (m_inputAdapter == nullptr)
    {
        m_inputAdapter = m_engine->createOwnedObject<WSClientInputAdapter>(
            // m_engine,
            type,
            pushMode,
            properties    
        );
    }
    return m_inputAdapter;
};

OutputAdapter* WSClientAdapterManager::getOutputAdapter()
{
    if (m_outputAdapter == nullptr)
    {
        m_outputAdapter = m_engine->createOwnedObject<WSClientOutputAdapter>(
            &m_client,
            &m_hdl,
            &m_active
        );
    }

    return m_outputAdapter;
}

DateTime WSClientAdapterManager::processNextSimTimeSlice( DateTime time )
{
    // no sim data
    return DateTime::NONE();
}

}