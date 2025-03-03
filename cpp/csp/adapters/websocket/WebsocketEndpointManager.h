#ifndef WEBSOCKET_ENDPOINT_MANAGER_H
#define WEBSOCKET_ENDPOINT_MANAGER_H

#include <boost/asio.hpp>
#include <csp/adapters/websocket/WebsocketEndpoint.h>
#include <csp/adapters/websocket/ClientAdapterManager.h>
#include <csp/adapters/websocket/ClientInputAdapter.h>
#include <csp/adapters/websocket/ClientOutputAdapter.h>
#include <csp/adapters/websocket/ClientHeaderUpdateAdapter.h>
#include <csp/adapters/websocket/ClientConnectionRequestAdapter.h>
#include <csp/core/Enum.h>
#include <csp/core/Hash.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/core/Platform.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <chrono>
#include <optional>
#include <functional>

namespace csp::adapters::websocket {
using namespace csp;
class WebsocketEndpoint;

class ClientAdapterManager;
class ClientOutputAdapter;
class ClientConnectionRequestAdapter;
class ClientHeaderUpdateOutputAdapter;

struct ConnectPayloads {
    std::vector<std::string> consumer_payloads;
    std::vector<std::string> producer_payloads;
};

struct EndpointConfig {
    std::chrono::milliseconds reconnect_interval;
    std::unique_ptr<boost::asio::steady_timer> reconnect_timer;
    bool attempting_reconnect{false};
    bool connected{false};
    
    // Payloads for different client types
    std::vector<std::string> consumer_payloads;
    std::vector<std::string> producer_payloads;

    explicit EndpointConfig(boost::asio::io_context& ioc) 
        : reconnect_timer(std::make_unique<boost::asio::steady_timer>(ioc)) {}
};

// Callbacks for endpoint events
struct EndpointCallbacks {
    std::function<void(const std::string&)> onOpen;
    std::function<void(const std::string&, const std::string&)> onFail;
    std::function<void(const std::string&)> onClose;
    std::function<void(const std::string&, const std::string&)> onSendFail;
    std::function<void(const std::string&, void*, size_t)> onMessage;
};

struct WebsocketClientStatusTypeTraits
{
    enum _enum : unsigned char
    {
        ACTIVE = 0,
        GENERIC_ERROR = 1,
        CONNECTION_FAILED = 2,
        CLOSED = 3,
        MESSAGE_SEND_FAIL = 4,

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using ClientStatusType = Enum<WebsocketClientStatusTypeTraits>;

class WebsocketEndpointManager {
public:
    explicit WebsocketEndpointManager(ClientAdapterManager* mgr, const Dictionary & properties, Engine* engine);
    ~WebsocketEndpointManager();
    void send(const std::string& value, const size_t& caller_id);
    // Whether the input adapter (subscribe) given by a specific caller_id was pruned
    bool adapterPruned( size_t caller_id );
    // Whether the output adapater (publish) given by a specific caller_id publishes to a given endpoint

    void start(DateTime starttime, DateTime endtime);
    void stop();

    void handleConnectionRequest(const Dictionary & properties, size_t validated_id, bool is_subscribe);
    void handleEndpointFailure(const std::string& endpoint_id, const std::string& reason, ClientStatusType status_type);
    
    void setupEndpoint(const std::string& endpoint_id, std::unique_ptr<WebsocketEndpoint> endpoint, std::string payload, bool persist, bool is_consumer, size_t validated_id);
    void shutdownEndpoint(const std::string& endpoint_id);

    void addConsumer(const std::string& endpoint_id, size_t caller_id);
    void addProducer(const std::string& endpoint_id, size_t caller_id);
    bool canRemoveEndpoint(const std::string& endpoint_id);

    void removeEndpointForCallerId(const std::string& endpoint_id, bool is_consumer, size_t validated_id);
    void removeConsumer(const std::string& endpoint_id, size_t caller_id);
    void removeProducer(const std::string& endpoint_id, size_t caller_id);

    WebsocketEndpoint * getNonDynamicEndpoint();
    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );
    OutputAdapter * getOutputAdapter( const Dictionary & properties );
    OutputAdapter * getHeaderUpdateAdapter();
    OutputAdapter * getConnectionRequestAdapter( const Dictionary & properties );
private:
    inline size_t validateCallerId(int64_t caller_id) const {
        if (caller_id < 0) {
            CSP_THROW(ValueError, "caller_id cannot be negative: " << caller_id);
        }
        return static_cast<size_t>(caller_id);
    }
    inline void ensureVectorSize(std::vector<bool>& vec, size_t caller_id) {
        if (vec.size() <= caller_id) {
            vec.resize(caller_id + 1, false);
        }
    }
    // Whether the output adapater (publish) given by a specific caller_id publishes to a given endpoint
    inline bool publishesToEndpoint(const size_t caller_id, const std::string& endpoint_id){
        auto config_it = m_endpoint_configs.find(endpoint_id); 
        if( config_it == m_endpoint_configs.end() || !config_it->second.connected )
            return false;

        return caller_id < m_endpoint_producers[endpoint_id].size() && 
        m_endpoint_producers[endpoint_id][caller_id];
    }
    size_t m_num_threads;
    net::io_context m_ioc;
    Engine* m_engine;
    boost::asio::strand<boost::asio::io_context::executor_type> m_strand;
    ClientAdapterManager* m_mgr;
    ClientHeaderUpdateOutputAdapter* m_updateAdapter;
    std::vector<std::unique_ptr<std::thread>> m_threads;
    Dictionary m_properties;
    std::vector<ClientConnectionRequestAdapter*> m_connectionRequestAdapters;

    // Bidirectional mapping using vectors since caller_ids are sequential
    // Maybe not efficient? Should be good for small number of edges though
    std::unordered_map<std::string, std::vector<bool>> m_endpoint_consumers;  // endpoint_id -> vector[caller_id] for consuemrs
    std::unordered_map<std::string, std::vector<bool>> m_endpoint_producers;  // endpoint_id -> vector[caller_id] for producers
    
    // Quick lookup for caller's endpoints
    std::vector< std::unordered_set<std::string> > m_consumer_endpoints;  // caller_id -> set of endpoints they consume from
    std::vector< std::unordered_set<std::string> > m_producer_endpoints;  // caller_id -> set of endpoints they produce to
    boost::asio::executor_work_guard<boost::asio::io_context::executor_type> m_work_guard;
    std::unordered_map<std::string, std::unique_ptr<WebsocketEndpoint>> m_endpoints;
    std::unordered_map<std::string, EndpointConfig> m_endpoint_configs;
    std::vector<ClientInputAdapter*> m_inputAdapters;
    std::vector<ClientOutputAdapter*> m_outputAdapters;
    bool m_dynamic;
};

}
#endif