#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_CONNECTIONREQUESTADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_CONNECTIONREQUESTADAPTER_H

#include <csp/adapters/websocket/ClientAdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>
#include <csp/adapters/websocket/csp_autogen/websocket_types.h>

namespace csp::adapters::websocket
{
using namespace csp::autogen;

class ClientAdapterManager;
class WebsocketEndpointManager;

class ClientConnectionRequestAdapter final: public OutputAdapter
{
public:
    ClientConnectionRequestAdapter(
        Engine * engine,
        WebsocketEndpointManager * websocketManager,
        bool isSubscribe,
        size_t callerId,
        boost::asio::strand<boost::asio::io_context::executor_type>& strand
    );

    void executeImpl() override;

    const char * name() const override { return "WebsocketClientConnectionRequestAdapter"; }

private:
    WebsocketEndpointManager* m_websocketManager;
    boost::asio::strand<boost::asio::io_context::executor_type>& m_strand;
    bool m_isSubscribe;
    size_t m_callerId;
    bool m_checkPerformed;
    bool m_isPruned{false};

};

}


#endif