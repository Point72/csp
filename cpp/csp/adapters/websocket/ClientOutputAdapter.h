#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_OUTPUTADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_OUTPUTADAPTER_H

#include <csp/adapters/websocket/WebsocketEndpoint.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>

namespace csp::adapters::websocket
{

class ClientAdapterManager;

class ClientOutputAdapter final: public OutputAdapter
{

public:
    ClientOutputAdapter(
        Engine * engine,
        WebsocketEndpoint& endpoint
    );

    void executeImpl() override;

    const char * name() const override { return "WebsocketClientOutputAdapter"; }

private:
    WebsocketEndpoint& m_endpoint;
};

}


#endif