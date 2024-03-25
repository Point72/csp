#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_ADAPTERMGR_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_ADAPTERMGR_H

#include <csp/core/Enum.h>
#include <csp/core/Hash.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PushInputAdapter.h>
#include <thread>

#include <csp/adapters/websockets/ClientInputAdapter.h>
#include <csp/adapters/websockets/ClientOutputAdapter.h>
#include <csp/adapters/websockets/ClientHeaderUpdateAdapter.h>
#include <csp/adapters/websockets/WebsocketEndpoint.h>

namespace csp::adapters::websockets {

using namespace csp;

struct WebsocketClientStatusTypeTraits
{
    enum _enum : unsigned char
    {
        ACTIVE = 0,
        GENERIC_ERROR = 1,
        CONNECTION_FAILED = 2,
        CLOSED = 3,

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using ClientStatusType = Enum<WebsocketClientStatusTypeTraits>;

class ClientAdapterManager final : public AdapterManager
{


public:
    ClientAdapterManager(
        Engine * engine,
        const Dictionary & properties
    );
    ~ClientAdapterManager();

    const char * name() const override { return "ClientAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;

    void stop() override;

    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );
    OutputAdapter * getOutputAdapter();
    OutputAdapter * getHeaderUpdateAdapter();

    DateTime processNextSimTimeSlice( DateTime time ) override;

private:
    // need some client info
    
    bool m_active;
    bool m_shouldRun;
    WebsocketEndpointBase* m_endpoint;
    ClientInputAdapter* m_inputAdapter;
    ClientOutputAdapter* m_outputAdapter;
    ClientHeaderUpdateAdapter* m_updateAdapter;
    std::unique_ptr<std::thread> m_thread;
    Dictionary m_properties;
};

}

#endif