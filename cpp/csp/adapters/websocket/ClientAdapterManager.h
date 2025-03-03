#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_ADAPTERMGR_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_ADAPTERMGR_H

#include <csp/adapters/websocket/WebsocketEndpoint.h>
#include <csp/adapters/websocket/WebsocketEndpointManager.h>
#include <csp/adapters/websocket/ClientInputAdapter.h>
#include <csp/adapters/websocket/ClientHeaderUpdateAdapter.h>
#include <csp/core/Enum.h>
#include <csp/core/Hash.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/core/Platform.h>
#include <thread>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <unordered_set>


namespace csp::adapters::websocket {

using namespace csp;

class WebsocketEndpointManager;

class ClientAdapterManager final : public AdapterManager
{


public:
    ClientAdapterManager(
        Engine * engine,
        const Dictionary & properties
    );
    ~ClientAdapterManager();

    const char * name() const override { return "WebsocketClientAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;

    void stop() override;

    WebsocketEndpointManager* getWebsocketManager();
    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );
    OutputAdapter * getOutputAdapter( const Dictionary & properties );
    OutputAdapter * getHeaderUpdateAdapter();
    OutputAdapter * getConnectionRequestAdapter( const Dictionary & properties );

    DateTime processNextSimTimeSlice( DateTime time ) override;

private:
    Dictionary m_properties;
    std::unique_ptr<WebsocketEndpointManager> m_endpointManager;
};

}

#endif