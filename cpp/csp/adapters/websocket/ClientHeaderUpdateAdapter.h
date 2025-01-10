#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_HEADERUPDATEADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_HEADERUPDATEADAPTER_H

#include <csp/adapters/websocket/WebsocketEndpointManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>
#include <csp/adapters/websocket/csp_autogen/websocket_types.h>

namespace csp::adapters::websocket
{
using namespace csp::autogen;

class WebsocketEndpointManager;

class ClientHeaderUpdateOutputAdapter final: public OutputAdapter
{
public:
    ClientHeaderUpdateOutputAdapter(
        Engine * engine,
        WebsocketEndpointManager * mgr,
        boost::asio::strand<boost::asio::io_context::executor_type>& strand
    );

    void executeImpl() override;

    const char * name() const override { return "WebsocketClientHeaderUpdateAdapter"; }

private:
    WebsocketEndpointManager * m_mgr;
    boost::asio::strand<boost::asio::io_context::executor_type>& m_strand;

    

};

}


#endif