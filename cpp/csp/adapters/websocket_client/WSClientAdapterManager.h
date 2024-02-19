#ifndef _IN_CSP_ADAPTERS_WSCLIENT_ADAPTERMGR_H
#define _IN_CSP_ADAPTERS_WSCLIENT_ADAPTERMGR_H

#include <csp/core/Enum.h>
#include <csp/core/Hash.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/adapters/websocket_client/WSClientInputAdapter.h>
#include <csp/adapters/websocket_client/WSClientOutputAdapter.h>
#include <thread>

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>

namespace csp::adapters::wsclient {

struct WebsocketClientStatusTypeTraits
{
    enum _enum : unsigned char
    {
        ACTIVE = 0,
        GENERIC_ERROR = 1,
        CONNECTION_FAILED = 2,

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using WSClientStatusType = csp::Enum<WebsocketClientStatusTypeTraits>;

typedef websocketpp::client<websocketpp::config::asio_client> client;

class WSClientAdapterManager final : public csp::AdapterManager
{
public:
    WSClientAdapterManager(
        csp::Engine * engine,
        const csp::Dictionary & properties
    );
    ~WSClientAdapterManager();

    const char * name() const override { return "WSClientAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;

    void stop() override;

    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );
    OutputAdapter * getOutputAdapter();

    DateTime processNextSimTimeSlice( DateTime time ) override;

private:
    // need some client info
    client m_client;
    WSClientInputAdapter* m_inputAdapter;
    websocketpp::connection_hdl m_hdl;
    WSClientOutputAdapter* m_outputAdapter;
    std::unique_ptr<std::thread> m_thread;
    bool m_threadActive;
    bool m_shouldRun;
    bool m_active;
    const Dictionary m_properties;

private:
    // callback to the input adapter processMessage
    void onMessage( websocketpp::connection_hdl, message_ptr msg);
    void onOpen( websocketpp::connection_hdl );
    void onFail( websocketpp::connection_hdl );

    void innerLoop();
};

}

#endif
