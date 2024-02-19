#ifndef _IN_CSP_ADAPTERS_WSCLIENT_OUTPUTADAPTER_H
#define _IN_CSP_ADAPTERS_WSCLIENT_OUTPUTADAPTER_H

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>

namespace csp::adapters::wsclient
{

typedef websocketpp::client<websocketpp::config::asio_client> client;

class WSClientOutputAdapter final: public OutputAdapter
{
public:
    WSClientOutputAdapter(
        Engine * engine,
        client* client,
        websocketpp::connection_hdl* hdl,
        bool* active
    );

    void executeImpl() override;

    const char * name() const override { return "WSClientOutputAdapter"; }

private:
    utils::OutputDataMapperPtr  m_dataMapper;
    client* m_client;
    websocketpp::connection_hdl* m_hdl;
    bool* m_active;
};
    
} // namespace csp::adapters::wsclient


#endif //_IN_CSP_ADAPTERS_WSCLIENT_OUTPUTADAPTER_H