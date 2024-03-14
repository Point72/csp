#include <csp/adapters/websockets/WebsocketEndpoint.h>

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

namespace csp::adapters::websockets {
using namespace csp;

/*
WebsocketEndpointBase -> base setup
*/
WebsocketEndpointBase::WebsocketEndpointBase(csp::Dictionary properties)
: m_properties(properties)
{ };

csp::Dictionary& WebsocketEndpointBase::getProperties() {
    return m_properties;
}

void WebsocketEndpointBase::setOnMessageCb(on_message_cb cb) 
{ m_on_message=cb; }

void WebsocketEndpointBase::setOnOpenCb(void_cb cb) 
{ m_on_open=cb; }

void WebsocketEndpointBase::setOnFailCb(void_cb cb) 
{ m_on_fail=cb; }

void WebsocketEndpointBase::setOnCloseCb(void_cb cb)
{ m_on_close=cb; }

void WebsocketEndpointBase::onOpen(websocketpp::connection_hdl)
{ m_on_open(); }
void WebsocketEndpointBase::onMessage(websocketpp::connection_hdl, message_ptr msg)
{ m_on_message(msg->get_payload()); }
void WebsocketEndpointBase::onFail(websocketpp::connection_hdl)
{ m_on_fail(); }
void WebsocketEndpointBase::onClose(websocketpp::connection_hdl)
{ m_on_close(); }

/*
WebsocketEndpointTLS -> tls impl
*/
WebsocketEndpointTLS::WebsocketEndpointTLS(csp::Dictionary properties)
: WebsocketEndpointBase(std::move(properties))
{
    if (m_properties.get<bool>("verbose_log")) {
        m_client.set_access_channels(websocketpp::log::alevel::all);
    } else {
        m_client.clear_access_channels(websocketpp::log::alevel::all);
    }
    m_client.init_asio();

    // need to set these with callbacks
    m_client.set_open_handler(bind(&WebsocketEndpointBase::onOpen, this, ::_1));
    m_client.set_message_handler(bind(&WebsocketEndpointBase::onMessage, this, ::_1, ::_2));
    m_client.set_fail_handler(bind(&WebsocketEndpointBase::onFail, this, ::_1));
    m_client.set_close_handler(bind(&WebsocketEndpointBase::onClose, this, ::_1));
    m_client.set_tls_init_handler([this](websocketpp::connection_hdl){
        auto ctx = websocketpp::lib::make_shared<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12);
        boost::system::error_code ec;
        ctx->set_options(
            boost::asio::ssl::context::default_workarounds |
            boost::asio::ssl::context::no_sslv2 |
            boost::asio::ssl::context::single_dh_use, ec
        );
        if (ec) {
            CSP_THROW(csp::RuntimeException, "Init tls failed: "<< ec);
        }
        return ctx;
    });
}

void WebsocketEndpointTLS::send(const std::string& s)
{
    websocketpp::lib::error_code ec;
    m_client.send(m_hdl, s, websocketpp::frame::opcode::value::TEXT, ec);
}

void WebsocketEndpointTLS::run()
{
    auto uri = m_properties.get<std::string>("uri");
    websocketpp::lib::error_code ec;
    tls_client::connection_ptr con = m_client.get_connection(uri, ec);
    const csp::Dictionary &headers = *m_properties.get<DictionaryPtr>("headers");

    for( auto it = headers.begin(); it != headers.end(); ++it )
    {
        const std::string key = it.key();
        const std::string value = headers.get<std::string>( key );
        con.get()->append_header(key, value);
    }
    if (ec) {
        CSP_THROW(RuntimeException, "could not create connection because: " << ec.message());
    }

    m_client.connect(con);
    m_hdl = con->get_handle();
    m_client.run();
    m_client.reset();
}

void WebsocketEndpointTLS::close()
{
    websocketpp::lib::error_code ec;
    m_client.close(m_hdl, websocketpp::close::status::going_away, "", ec);
    if (ec) {
        CSP_THROW(RuntimeException, "could not close connection because: " << ec.message());
    }
}


/*
WebsocketEndpointNoTLS -> tls impl
*/
WebsocketEndpointNoTLS::WebsocketEndpointNoTLS(csp::Dictionary properties)
: WebsocketEndpointBase(properties) 
{
    if (m_properties.get<bool>("verbose_log")) {
        m_client.set_access_channels(websocketpp::log::alevel::all);
        // m_client.clear_access_channels(websocketpp::log::alevel::frame_payload);
    } else {
        m_client.clear_access_channels(websocketpp::log::alevel::all);
    }
    m_client.init_asio();

    // need to set these with callbacks
    m_client.set_open_handler(bind(&WebsocketEndpointBase::onOpen, this, ::_1));
    m_client.set_message_handler(bind(&WebsocketEndpointBase::onMessage, this, ::_1, ::_2));
    m_client.set_fail_handler(bind(&WebsocketEndpointBase::onFail, this, ::_1));
    m_client.set_close_handler(bind(&WebsocketEndpointBase::onClose, this, ::_1));
}

void WebsocketEndpointNoTLS::send(const std::string& s)
{
    websocketpp::lib::error_code ec;
    m_client.send(m_hdl, s, websocketpp::frame::opcode::value::TEXT, ec);
}

void WebsocketEndpointNoTLS::run()
{
    auto uri = m_properties.get<std::string>("uri");
    websocketpp::lib::error_code ec;
    client::connection_ptr con = m_client.get_connection(uri, ec);
    const csp::Dictionary &headers = *m_properties.get<DictionaryPtr>("headers");

    for( auto it = headers.begin(); it != headers.end(); ++it )
    {
        const std::string key = it.key();
        const std::string value = headers.get<std::string>( key );
        con.get()->append_header(key, value);
    }
    if (ec) {
        CSP_THROW(RuntimeException, "could not create connection because: " << ec.message());
    }

    m_client.connect(con);
    m_hdl = con->get_handle();
    m_client.run();
    m_client.reset();
}

void WebsocketEndpointNoTLS::close()
{
    websocketpp::lib::error_code ec;
    m_client.close(m_hdl, websocketpp::close::status::going_away, "Good bye", ec);
    if (ec) {
        CSP_THROW(RuntimeException, "could not close connection because: " << ec.message());
    }
}

}