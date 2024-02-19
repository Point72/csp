#ifndef _IN_CSP_ADAPTERS_WSCLIENT_INPUTADAPTER_H
#define _IN_CSP_ADAPTERS_WSCLIENT_INPUTADAPTER_H

#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <csp/engine/Dictionary.h>
#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/engine/Struct.h>

namespace csp::adapters::wsclient
{

typedef websocketpp::config::asio_client::message_type::ptr message_ptr;


class WSClientInputAdapter final: public PushInputAdapter
{
public:
    WSClientInputAdapter(
        Engine * engine,
        CspTypePtr & type,
        PushMode pushMode,
        const csp::Dictionary & properties
    );

    void processMessage( message_ptr message, csp::PushBatch* batch );

private:
    utils::MessageStructConverterPtr m_converter;

};
    
} // namespace csp::adapters::wsclient


#endif //_IN_CSP_ADAPTERS_WSCLIENT_INPUTADAPTER_H