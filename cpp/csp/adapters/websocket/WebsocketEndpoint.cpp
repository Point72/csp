#include <csp/adapters/websocket/WebsocketEndpoint.h>

namespace csp::adapters::websocket {
using namespace csp;

WebsocketEndpoint::WebsocketEndpoint(
    net::io_context& ioc,
    Dictionary properties 
) : m_properties(std::make_shared<Dictionary>(std::move(properties))),
    m_ioc(ioc)
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
    // Owns this ioc object
    if(m_properties->get<bool>("use_ssl")) {
        ssl::context ctx{ssl::context::sslv23};
        ctx.set_verify_mode(ssl::context::verify_peer );
        ctx.set_default_verify_paths();

        m_session = std::make_shared<WebsocketSessionTLS>(
            m_ioc,
            ctx,
            m_properties,
            m_on_open, 
            m_on_fail, 
            m_on_message, 
            m_on_close, 
            m_on_send_fail
        );
    } else {
        m_session = std::make_shared<WebsocketSessionNoTLS>(
            m_ioc, 
            m_properties,
            m_on_open, 
            m_on_fail, 
            m_on_message, 
            m_on_close, 
            m_on_send_fail
        );
    }
    m_session->run();
}

WebsocketEndpoint::~WebsocketEndpoint() {
    try {
        // Call stop but explicitly pass false to prevent io_context shutdown
        stop(false);
    } catch (...) {
        // Ignore any exceptions during cleanup
    }
}

void WebsocketEndpoint::stop( bool stop_ioc )
{
    if( m_session ) m_session->stop(); 
    if( stop_ioc ) m_ioc.stop();
}

void WebsocketEndpoint::updateHeaders(csp::Dictionary properties){
    DictionaryPtr headers = m_properties->get<DictionaryPtr>("headers");
    for (auto it = properties.begin(); it != properties.end(); ++it) {
        std::string key = it.key();
        auto value = it.value<std::string>();
        headers->update(key, std::move(value));
    }
}

std::shared_ptr<Dictionary> WebsocketEndpoint::getProperties() {
    return m_properties;
}

void WebsocketEndpoint::send(const std::string& s)
{ if(m_session) m_session->send(s); }
void WebsocketEndpoint::ping()
{ if(m_session) m_session->ping(); }

}