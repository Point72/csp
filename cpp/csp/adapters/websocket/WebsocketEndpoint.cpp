#include <csp/adapters/websocket/WebsocketEndpoint.h>

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

namespace csp::adapters::websocket {
using namespace csp;

/*
WebsocketEndpointBase -> base setup
*/
WebsocketEndpointBase::WebsocketEndpointBase( csp::Dictionary properties )
: m_properties( properties )
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

void WebsocketEndpointBase::setOnSendFailCb(on_send_fail_cb cb)
{ m_on_send_fail=cb; }

/*
WebsocketEndpointTLS -> tls impl
*/
WebsocketEndpointTLS::WebsocketEndpointTLS(csp::Dictionary properties)
: WebsocketEndpointBase( std::move(properties) )
{
    if( m_properties.get<bool>("verbose_log") ) {
        m_client.set_access_channels(websocketpp::log::alevel::all);
    } else {
        m_client.clear_access_channels(websocketpp::log::alevel::all);
    }
    m_client.init_asio();

    // need to set these with callbacks
    m_client.set_open_handler( [ this ]( websocketpp::connection_hdl ){
        m_on_open();
    });
    m_client.set_message_handler( [ this ]( websocketpp::connection_hdl, message_ptr msg ) {
        m_on_message( msg -> get_payload() );
    });
    m_client.set_fail_handler( [ this ]( websocketpp::connection_hdl ){
        m_on_fail();
    });
    m_client.set_close_handler( [ this ]( websocketpp::connection_hdl ){
        m_on_close();
    });
    m_client.set_tls_init_handler( []( websocketpp::connection_hdl ){
        auto ctx = websocketpp::lib::make_shared<boost::asio::ssl::context>(boost::asio::ssl::context::tlsv12);
        boost::system::error_code ec;
        ctx->set_options(
            boost::asio::ssl::context::default_workarounds |
            boost::asio::ssl::context::no_sslv2 |
            boost::asio::ssl::context::single_dh_use, ec
        );
        if( ec ) {
            CSP_THROW( csp::RuntimeException, "Init tls failed: "<< ec );
        }
        return ctx;
    });
}
WebsocketEndpointTLS::~WebsocketEndpointTLS()
{ }

void WebsocketEndpointTLS::send( const std::string& s )
{
    websocketpp::lib::error_code ec;
    m_client.send( m_hdl, s, websocketpp::frame::opcode::value::TEXT, ec );
    if( ec ) m_on_send_fail(s);
}

void WebsocketEndpointTLS::run()
{
    auto uri = m_properties.get<std::string>("uri");
    websocketpp::lib::error_code ec;
    tls_client::connection_ptr con = m_client.get_connection( uri, ec );
    if( ec ) {
        CSP_THROW(RuntimeException, "could not create connection because: " << ec.message());
    }
    const csp::Dictionary &headers = *m_properties.get<DictionaryPtr>("headers");

    for( auto it = headers.begin(); it != headers.end(); ++it )
    {
        const std::string key = it.key();
        const std::string value = headers.get<std::string>( key );
        con.get() -> append_header( key, value );
    }

    m_client.connect( con );
    m_hdl = con -> get_handle();
    m_client.run();
    m_client.reset();
}

void WebsocketEndpointTLS::close()
{
    websocketpp::lib::error_code ec;
    m_client.close( m_hdl , websocketpp::close::status::going_away, "", ec );
    if( ec ) {
        CSP_THROW( RuntimeException, "could not close connection because: " << ec.message() );
    }
}


/*
WebsocketEndpointNoTLS -> tls impl
*/
WebsocketEndpointNoTLS::WebsocketEndpointNoTLS( csp::Dictionary properties )
: WebsocketEndpointBase( properties ) 
{
    if (m_properties.get<bool>("verbose_log")) {
        m_client.set_access_channels(websocketpp::log::alevel::all);
        // m_client.clear_access_channels(websocketpp::log::alevel::frame_payload);
    } else {
        m_client.clear_access_channels(websocketpp::log::alevel::all);
    }
    m_client.init_asio();

    // need to set these with callbacks
    m_client.set_open_handler( [ this ]( websocketpp::connection_hdl ){
        m_on_open();
    });
    m_client.set_message_handler( [ this ]( websocketpp::connection_hdl, message_ptr msg ) {
        m_on_message( msg -> get_payload() );
    });
    m_client.set_fail_handler( [ this ]( websocketpp::connection_hdl ){
        m_on_fail();
    });
    m_client.set_close_handler( [ this ]( websocketpp::connection_hdl ){
        m_on_close();
    });
}
WebsocketEndpointNoTLS::~WebsocketEndpointNoTLS()
{ }

void WebsocketEndpointNoTLS::send( const std::string& s )
{
    websocketpp::lib::error_code ec;
    m_client.send( m_hdl, s, websocketpp::frame::opcode::value::TEXT, ec );
    if( ec ) m_on_send_fail( s );
}

void WebsocketEndpointNoTLS::run()
{
    auto uri = m_properties.get<std::string>( "uri" );
    websocketpp::lib::error_code ec;
    client::connection_ptr con = m_client.get_connection( uri, ec );
    if(ec) {
        CSP_THROW( RuntimeException, "could not create connection because: " << ec.message() );
    }
    const csp::Dictionary &headers = *m_properties.get<DictionaryPtr>("headers");

    for( auto it = headers.begin(); it != headers.end(); ++it )
    {
        const std::string key = it.key();
        const std::string value = headers.get<std::string>( key );
        con.get() -> append_header( key, value );
    }

    m_client.connect( con );
    m_hdl = con -> get_handle();
    m_client.run();
    m_client.reset();
}

void WebsocketEndpointNoTLS::close()
{
    websocketpp::lib::error_code ec;
    m_client.close( m_hdl, websocketpp::close::status::going_away, "Good bye", ec );
    if( ec ) {
        CSP_THROW( RuntimeException, "could not close connection because: " << ec.message() );
    }
}

}