#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_OUTPUTADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_OUTPUTADAPTER_H

#include <csp/adapters/websocket/WebsocketEndpoint.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>
#include <csp/adapters/websocket/ClientAdapterManager.h> 

namespace csp::adapters::websocket
{

class ClientAdapterManager;
class WebsocketEndpointManager;

class ClientOutputAdapter final: public OutputAdapter
{

public:
    ClientOutputAdapter(
        Engine * engine,
        WebsocketEndpointManager * websocketManager,
        size_t caller_id,
        net::io_context& ioc,
        boost::asio::strand<boost::asio::io_context::executor_type>& strand
        // bool dynamic
    );

    void executeImpl() override;

    const char * name() const override { return "WebsocketClientOutputAdapter"; }

private:
    WebsocketEndpointManager* m_websocketManager;
    size_t m_callerId;
    [[maybe_unused]] net::io_context& m_ioc;
    boost::asio::strand<boost::asio::io_context::executor_type>& m_strand;
    // bool m_dynamic;
    // std::unordered_map<std::string, std::vector<bool>>& m_endpoint_consumers;
};

}


#endif