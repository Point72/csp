#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_OUTPUTADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_OUTPUTADAPTER_H

#include <csp/adapters/websockets/WebsocketEndpoint.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>
#include <websocketpp/client.hpp>

namespace csp::adapters::websockets
{


class ClientOutputAdapter final: public OutputAdapter
{

public:
    ClientOutputAdapter(
        Engine * engine,
        WebsocketEndpointBase* endpoint
    );

    void executeImpl() override;

    const char * name() const override { return "ClientOutputAdapter"; }

private:
    WebsocketEndpointBase* m_endpoint;
    websocketpp::connection_hdl* m_hdl;
    bool* m_active;
};

}


#endif