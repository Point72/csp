#include <csp/adapters/websocket/WebsocketEndpointManager.h>

namespace csp {

INIT_CSP_ENUM( adapters::websocket::ClientStatusType,
               "ACTIVE",
               "GENERIC_ERROR",
               "CONNECTION_FAILED",
               "CLOSED",
               "MESSAGE_SEND_FAIL",
);

}
namespace csp::adapters::websocket {

WebsocketEndpointManager::WebsocketEndpointManager( ClientAdapterManager* mgr, const Dictionary & properties, Engine* engine ) 
:   m_num_threads( static_cast<size_t>(properties.get<int64_t>("num_threads")) ),
    m_ioc( m_num_threads ),
    m_engine( engine ),
    m_strand( boost::asio::make_strand(m_ioc) ),
    m_mgr( mgr ),
    m_updateAdapter( nullptr ),
    m_properties( properties ),
    m_work_guard(boost::asio::make_work_guard(m_ioc)),
    m_dynamic( properties.get<bool>("dynamic") ){
    // Total number of subscribe and send function calls, set on the adapter manager
    // when is it created. Note, that some of the input adapters might have been
    // pruned from the graph and won't get created.
    auto input_size = static_cast<size_t>(properties.get<int64_t>("subscribe_calls"));
    m_inputAdapters.resize(input_size, nullptr);
    m_consumer_endpoints.resize(input_size);
    // send_calls
    auto output_size = static_cast<size_t>(properties.get<int64_t>("send_calls"));
    m_outputAdapters.resize(output_size, nullptr);
    m_producer_endpoints.resize(output_size);

    // We choose to not automatically size m_connectionRequestAdapters
    // since the index there is not meaningful,
    // producers and subscribers are combined.
    // We just hold onto their pointers.
};

WebsocketEndpointManager::~WebsocketEndpointManager()
{
}

void WebsocketEndpointManager::start(DateTime starttime, DateTime endtime) {
    m_ioc.reset();
    if( !m_dynamic ){
        boost::asio::post(m_strand, [this]() {
            // We subscribe for both the subscribe and send calls
            // But we probably should check here.
            if( m_outputAdapters.size() == 1)
                handleConnectionRequest(Dictionary(m_properties), 0, false);
            // If we have an input adapter call AND it's not pruned.
            if( m_inputAdapters.size() == 1 && !adapterPruned(0))
                handleConnectionRequest(Dictionary(m_properties), 0, true);
        });
    }
    for (size_t i = 0; i < m_num_threads; ++i) {
        m_threads.emplace_back(std::make_unique<std::thread>([this]() {
            m_ioc.run();
        }));
    }
};

bool WebsocketEndpointManager::adapterPruned( size_t caller_id ){
    return m_inputAdapters[caller_id] == nullptr;
};

void WebsocketEndpointManager::send(const std::string& value, const size_t& caller_id) {
    const auto& endpoints = m_producer_endpoints[caller_id];
    // For each endpoint this producer is connected to
    for (const auto& endpoint_id : endpoints) {
        // Double check the endpoint exists and producer is still valid
        if(publishesToEndpoint(caller_id, endpoint_id)) {
            auto it = m_endpoints.find(endpoint_id);
            if( it != m_endpoints.end())
                it->second.get()->send(value);
        }
    }
};

void WebsocketEndpointManager::removeEndpointForCallerId(const std::string& endpoint_id, bool is_consumer, size_t validated_id)
{
    if (is_consumer) {
        WebsocketEndpointManager::removeConsumer(endpoint_id, validated_id);
    } else {
        WebsocketEndpointManager::removeProducer(endpoint_id, validated_id);
    }
    if (canRemoveEndpoint(endpoint_id))
        shutdownEndpoint(endpoint_id);
}

void WebsocketEndpointManager::shutdownEndpoint(const std::string& endpoint_id) {
    // This functions should only be called from the thread running m_ioc
    // Cancel any pending reconnection attempts
    if (auto config_it = m_endpoint_configs.find(endpoint_id); 
        config_it != m_endpoint_configs.end()) {
        config_it->second.reconnect_timer->cancel();
        m_endpoint_configs.erase(config_it);
    }
    
    // Stop and remove the endpoint
    // No need to stop, destructo handles it
    if (auto endpoint_it = m_endpoints.find(endpoint_id); endpoint_it != m_endpoints.end())
        m_endpoints.erase(endpoint_it);
    std::stringstream ss;
    ss << "No more connections for endpoint={" << endpoint_id << "} Shutting down...";
    std::string msg = ss.str();
    m_mgr -> pushStatus(StatusLevel::INFO,  ClientStatusType::CLOSED, msg);
}

void WebsocketEndpointManager::setupEndpoint(const std::string& endpoint_id, 
                                            std::unique_ptr<WebsocketEndpoint> endpoint,
                                            std::string payload, 
                                            bool persist,
                                            bool is_consumer,
                                            size_t validated_id) 
{
    // Store the endpoint first
    auto& stored_endpoint = m_endpoints[endpoint_id] = std::move(endpoint);

    stored_endpoint->setOnOpen([this, endpoint_id, endpoint = stored_endpoint.get(), payload=std::move(payload), persist, is_consumer, validated_id]() {
        auto [iter, inserted] = m_endpoint_configs.try_emplace(endpoint_id, m_ioc);
        auto& config = iter->second;
        config.connected = true;
        config.attempting_reconnect = false;

        // Send consumer payloads
        const auto& consumers = m_endpoint_consumers[endpoint_id];
        for (size_t i = 0; i < config.consumer_payloads.size(); ++i) {
            if (!config.consumer_payloads[i].empty() && 
                i < consumers.size() && consumers[i]) {
                endpoint->send(config.consumer_payloads[i]);
            }
        }
        
        // Send producer payloads
        const auto& producers = m_endpoint_producers[endpoint_id];
        for (size_t i = 0; i < config.producer_payloads.size(); ++i) {
            if (!config.producer_payloads[i].empty() && 
                i < producers.size() && producers[i]) {
                endpoint->send(config.producer_payloads[i]);
            }
        }
        // should only happen if persist is False
        if ( !payload.empty() )
            endpoint -> send(payload);
        std::stringstream ss;
        ss << "Connected successfully for endpoint={" << endpoint_id << "}";
        std::string msg = ss.str();
        m_mgr -> pushStatus(StatusLevel::INFO, ClientStatusType::ACTIVE, msg);
        // We remove the caller id, if it was the only one, then we shut down the endpoint
        if( !persist )
            removeEndpointForCallerId(endpoint_id, is_consumer, validated_id);
    });

    stored_endpoint->setOnFail([this, endpoint_id](const std::string& reason) {
        handleEndpointFailure(endpoint_id, reason, ClientStatusType::CONNECTION_FAILED);
    });

    stored_endpoint->setOnClose([this, endpoint_id]() {
        // If we didn't close it ourselves
        if (auto config_it = m_endpoint_configs.find(endpoint_id); config_it != m_endpoint_configs.end())
            handleEndpointFailure(endpoint_id, "Connection closed", ClientStatusType::CLOSED);
    });
    stored_endpoint->setOnMessage([this, endpoint_id](void* data, size_t len) {
        // Here we need to route to all active consumers for this endpoint
        const auto& consumers = m_endpoint_consumers[endpoint_id];
        
        // For each active consumer, we need to send to their input adapter
        PushBatch batch( m_engine -> rootEngine() );  // TODO is this right?
        for (size_t consumer_id = 0; consumer_id < consumers.size(); ++consumer_id) {
                        if (consumers[consumer_id]) {
                std::vector<uint8_t> data_copy(static_cast<uint8_t*>(data), 
                                    static_cast<uint8_t*>(data) + len);
                auto tup = std::tuple<std::string, void*>{endpoint_id, data_copy.data()};
                m_inputAdapters[consumer_id] -> processMessage( endpoint_id, data_copy.data(), len, &batch );
            }
        }
    });
    stored_endpoint -> setOnSendFail(
        [ this, endpoint_id ]( const std::string& s ) {
            std::stringstream ss;
            ss << "Error: " << s << " for endpoint={" << endpoint_id << "}";
            std::string msg = ss.str();
            m_mgr -> pushStatus( StatusLevel::ERROR, ClientStatusType::MESSAGE_SEND_FAIL, msg );
        }
    );
    stored_endpoint -> run();
};


void WebsocketEndpointManager::handleEndpointFailure(const std::string& endpoint_id, 
                                               const std::string& reason, ClientStatusType status_type) {
    // If there are any active consumers/producers, try to reconnect
    if (!canRemoveEndpoint(endpoint_id)) {
        auto [iter, inserted] = m_endpoint_configs.try_emplace(endpoint_id, m_ioc);
        auto& config = iter->second;
        config.connected = false;
        
        if (!config.attempting_reconnect) {
            config.attempting_reconnect = true;
            
            // Schedule reconnection attempt
            config.reconnect_timer->expires_after(config.reconnect_interval);
            config.reconnect_timer->async_wait([this, endpoint_id](const error_code& ec) {
                // boost::asio::post(m_ioc, [this, endpoint_id]() {
                // If we still want to subscribe to this endpoint
                if (auto it = m_endpoints.find(endpoint_id); 
                    it != m_endpoints.end()) {
                    auto config_it = m_endpoint_configs.find(endpoint_id);
                    if (config_it != m_endpoint_configs.end()) {
                        auto& config = config_it -> second;
                        // We are no longer attempting to reconnect
                        config.attempting_reconnect = false;
                    }
                    it->second->run();  // Attempt to reconnect
                }
            });
        }
    } else {
        // No active consumers/producers, clean up the endpoint
        m_endpoints.erase(endpoint_id);
        m_endpoint_configs.erase(endpoint_id);
    }
    
    std::stringstream ss;
    ss << "Connection Failure for endpoint={" << endpoint_id << "} Due to: " << reason;
    std::string msg = ss.str();
    if ( status_type == ClientStatusType::CLOSED || status_type == ClientStatusType::ACTIVE )
        m_mgr -> pushStatus(StatusLevel::INFO, status_type, msg);
    else{
       m_mgr -> pushStatus(StatusLevel::ERROR, status_type, msg);
    }
};

void WebsocketEndpointManager::handleConnectionRequest(const Dictionary & properties, size_t validated_id, bool is_subscribe)
{
    // This should only get called from the thread running
    // m_ioc. This allows us to avoid locks on internal data
    // structures
    auto endpoint_id = properties.get<std::string>("uri");
    autogen::ActionType action = autogen::ActionType::create( properties.get<std::string>("action") );
    switch(action.enum_value()) {
        case autogen::ActionType::enum_::CONNECT: {
            auto persistent = properties.get<bool>("persistent");
            auto reconnect_interval = properties.get<TimeDelta>("reconnect_interval");
            // Update endpoint config
            auto& config = m_endpoint_configs.try_emplace(endpoint_id, m_ioc).first->second;

            config.reconnect_interval = std::chrono::milliseconds(
                reconnect_interval.asMilliseconds()
            );
            std::string payload = "";
            bool has_payload = properties.tryGet<std::string>("on_connect_payload", payload);

            if (has_payload && !payload.empty() && persistent) {
                auto& payloads = is_subscribe ? config.consumer_payloads : config.producer_payloads;
                if (payloads.size() <= validated_id) {
                    payloads.resize(validated_id + 1);
                }
                payloads[validated_id] = std::move(payload);  // Move to config
            }

            if ( persistent ){
                if (is_subscribe) {
                    WebsocketEndpointManager::addConsumer(endpoint_id, validated_id);
                } else {
                    WebsocketEndpointManager::addProducer(endpoint_id, validated_id);
                }
            }

            bool is_new_endpoint = !m_endpoints.contains(endpoint_id);
            if (is_new_endpoint) {
                auto endpoint = std::make_unique<WebsocketEndpoint>(m_ioc, properties);
                // We can safely move payload regardless - if it was never written to, it's just an empty string
                WebsocketEndpointManager::setupEndpoint(endpoint_id, std::move(endpoint), 
                                                    (has_payload && !payload.empty() && persistent) ? "" : std::move(payload),
                                                    persistent, is_subscribe, validated_id );
            }
            else{
                if( !persistent && !payload.empty() )
                    m_endpoints[endpoint_id]->send(payload);
                // Conscious decision to let non-persisten connection
                // results to update the header
                auto headers = properties.get<std::string>("headers");
                m_endpoints[endpoint_id]->updateHeaders(std::move(headers));
            }
            break;
        }
        
        case csp::autogen::ActionType::enum_::DISCONNECT: {
            // Clear persistence flag for this caller
            removeEndpointForCallerId(endpoint_id, is_subscribe, validated_id);
            break;
        }
        
        case csp::autogen::ActionType::enum_::PING: {
            // Only ping if the caller is actually connected to this endpoint
            auto& consumers = m_endpoint_consumers[endpoint_id];
            auto& producers = m_endpoint_producers[endpoint_id];

            if ( ( is_subscribe && validated_id < consumers.size() && consumers[validated_id] ) || 
                ( !is_subscribe && validated_id < producers.size() && producers[validated_id] ) ) {
                    if (auto it = m_endpoints.find(endpoint_id); it != m_endpoints.end()) {
                        it->second.get()->ping();
                    }
            }
            break;
        }
    }
};

WebsocketEndpoint * WebsocketEndpointManager::getNonDynamicEndpoint(){
    // Should only be called if dynamic = False
    if (!m_endpoints.empty()) {
        return m_endpoints.begin()->second.get();
    }
    return nullptr;
}

void WebsocketEndpointManager::addConsumer(const std::string& endpoint_id, size_t caller_id) {
    ensureVectorSize(m_endpoint_consumers[endpoint_id], caller_id);
    m_endpoint_consumers[endpoint_id][caller_id] = true;

    m_consumer_endpoints[caller_id].insert(endpoint_id);
};

void WebsocketEndpointManager::addProducer(const std::string& endpoint_id, size_t caller_id) {
    ensureVectorSize(m_endpoint_producers[endpoint_id], caller_id);
    m_endpoint_producers[endpoint_id][caller_id] = true;

    m_producer_endpoints[caller_id].insert(endpoint_id);
};

bool WebsocketEndpointManager::canRemoveEndpoint(const std::string& endpoint_id) {
    const auto& consumers = m_endpoint_consumers[endpoint_id];
    const auto& producers = m_endpoint_producers[endpoint_id];
    
    // Check if any true values exist in either vector
    return std::none_of(consumers.begin(), consumers.end(), [](bool b) { return b; }) &&
            std::none_of(producers.begin(), producers.end(), [](bool b) { return b; });
};

void WebsocketEndpointManager::removeConsumer(const std::string& endpoint_id, size_t caller_id) {
    auto& consumers = m_endpoint_consumers[endpoint_id];
    // Possibility it might not be subscribed,
    // so we have this check.
    if (caller_id < consumers.size()) {
        consumers[caller_id] = false;
    }
    // We initialize these upfront, this will be valid.
    m_consumer_endpoints[caller_id].erase(endpoint_id);
};

void WebsocketEndpointManager::removeProducer(const std::string& endpoint_id, size_t caller_id) {
    auto& producers = m_endpoint_producers[endpoint_id];
    // Possibility it might not be publihsing to
    // so we have this check.
    if (caller_id < producers.size()) {
        producers[caller_id] = false;
    }
    
    // We initialize these upfront, this will be valid.
    m_producer_endpoints[caller_id].erase(endpoint_id);
};


void WebsocketEndpointManager::stop() {
    // Stop all endpoints
    // Endpoints running on m_ioc thread,
    // So we call stop there
    boost::asio::post(m_strand, [this]() {
        for (auto& [endpoint_id, _] : m_endpoints) {
            shutdownEndpoint(endpoint_id);
        }
    });
    // Stop the work guard to allow the io_context to complete
    m_work_guard.reset();
    m_ioc.stop();
    
    // Wait for all threads to finish
    for (auto& thread : m_threads) {
        if (thread && thread->joinable()) {
            thread->join();
        }
    }
    
    // Clear threads before other members are destroyed
    m_threads.clear();
};

PushInputAdapter* WebsocketEndpointManager::getInputAdapter(CspTypePtr & type, PushMode pushMode, const Dictionary & properties)
{   
    auto caller_id = properties.get<int64_t>("caller_id");
    size_t validated_id = validateCallerId(caller_id);
    auto input_adapter = m_engine -> createOwnedObject<ClientInputAdapter>(
        type,
        pushMode,
        properties,
        m_dynamic
    );
    m_inputAdapters[validated_id] = input_adapter;
    return m_inputAdapters[validated_id];
};

OutputAdapter* WebsocketEndpointManager::getOutputAdapter( const Dictionary & properties )
{
    auto caller_id = properties.get<int64_t>("caller_id");
    size_t validated_id = validateCallerId(caller_id);
    assert(!properties.get<bool>("is_subscribe"));
    assert(m_outputAdapters.size() == validated_id);

    auto output_adapter = m_engine -> createOwnedObject<ClientOutputAdapter>( this, validated_id, m_ioc, m_strand );
    m_outputAdapters[validated_id] = output_adapter;
    return m_outputAdapters[validated_id];
};

OutputAdapter * WebsocketEndpointManager::getHeaderUpdateAdapter()
{
    if (m_updateAdapter == nullptr)
        m_updateAdapter = m_engine -> createOwnedObject<ClientHeaderUpdateOutputAdapter>( this, m_strand );

    return m_updateAdapter;
};

OutputAdapter * WebsocketEndpointManager::getConnectionRequestAdapter( const Dictionary & properties )
{
    auto caller_id = properties.get<int64_t>("caller_id");
    auto is_subscribe = properties.get<bool>("is_subscribe");
    
    auto* adapter = m_engine->createOwnedObject<ClientConnectionRequestAdapter>(
        this, is_subscribe, caller_id, m_strand
    );
    m_connectionRequestAdapters.push_back(adapter);
    
    return adapter;
};

}
