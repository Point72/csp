#include <csp/adapters/websocket/WebsocketEndpoint.h>

namespace csp::adapters::websocket {
using namespace csp;

WebsocketEndpoint::WebsocketEndpoint( 
    Dictionary properties 
) : m_properties(properties)
{ };
void WebsocketEndpoint::setOnOpen(void_cb on_open)
{ m_on_open = std::move(on_open); }
void WebsocketEndpoint::setOnFail(string_cb on_fail)
{ m_on_fail = std::move(on_fail); }
void WebsocketEndpoint::setOnMessage(char_cb on_message)
{ m_on_message = std::move(on_message); }
void WebsocketEndpoint::setOnClose(void_cb on_close)
{ m_on_close = std::move(on_close); }
void WebsocketEndpoint::setOnSendFail(string_cb on_send_fail)
{ m_on_send_fail = std::move(on_send_fail); }

void WebsocketEndpoint::run()
{

    m_ioc.restart();
    if(m_properties.get<bool>("use_ssl")) {
        ssl::context ctx{ssl::context::sslv23};
        ctx.set_verify_mode(ssl::context::verify_peer );
        ctx.set_default_verify_paths();

        m_session = new WebsocketSessionTLS(
            m_ioc,
            ctx,
            &m_properties,
            m_on_open, 
            m_on_fail, 
            m_on_message, 
            m_on_close, 
            m_on_send_fail
        );
    } else {
        m_session = new WebsocketSessionNoTLS(
            m_ioc, 
            &m_properties,
            m_on_open, 
            m_on_fail, 
            m_on_message, 
            m_on_close, 
            m_on_send_fail
        );
    }
    m_session->run();

    m_ioc.run();
}

void WebsocketEndpoint::stop()
{ 
    m_ioc.stop();
    if(m_session) m_session->stop(); 
}


csp::Dictionary& WebsocketEndpoint::getProperties() {
    return m_properties;
}

void WebsocketEndpoint::send(const std::string& s)
{ if(m_session) m_session->send(s); }


}