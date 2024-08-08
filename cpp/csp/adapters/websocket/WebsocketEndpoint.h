#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_ENDPOINT_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_ENDPOINT_H

#include <boost/beast/core.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/asio/strand.hpp>

#include <csp/engine/Dictionary.h>
#include <csp/core/Exception.h>

#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>

namespace csp::adapters::websocket {
using namespace csp;

namespace beast = boost::beast;         // from <boost/beast.hpp>
namespace http = beast::http;           // from <boost/beast/http.hpp>
namespace net = boost::asio;            // from <boost/asio.hpp>
namespace ssl = boost::asio::ssl;       // from <boost/asio/ssl.hpp>
namespace websocket = beast::websocket; // from <boost/beast/websocket.hpp>
using tcp = boost::asio::ip::tcp;       // from <boost/asio/ip/tcp.hpp>

using string_cb = std::function<void(const std::string&)>;
using char_cb = std::function<void(void*, size_t)>;
using void_cb = std::function<void()>;

class BaseWebsocketSession {
public:
    virtual void stop() { };
    virtual void send( const std::string& ) { };
    virtual void do_read() { };
    virtual void do_write(const std::string& ) { };
    virtual void run() { };
};

template<class Derived>
class WebsocketSession : public BaseWebsocketSession {
public:
    WebsocketSession(
        net::io_context& ioc,
        Dictionary* properties,
        void_cb& on_open,
        string_cb& on_fail,
        char_cb& on_message,
        void_cb& on_close,
        string_cb& on_send_fail
    ) : m_resolver( net::make_strand( ioc ) ),
        m_properties( properties ), 
        m_on_open( on_open ),
        m_on_fail( on_fail ),
        m_on_message( on_message ),
        m_on_close( on_close ),
        m_on_send_fail( on_send_fail )
    { };

    Derived& derived(){ return static_cast<Derived&>(*this); }

    void handle_message( beast::error_code& ec, std::size_t& bytes_transfered ) {
        if( ec ) {
            m_on_close();
            return;
        }
        m_on_message(beast::buffers_front(m_buffer.data()).data(), m_buffer.size());
        m_buffer.consume(m_buffer.size());
        do_read();
    }

    void set_headers(websocket::request_type& req) {
        const csp::Dictionary &headers = *m_properties->get<DictionaryPtr>("headers");

        for( auto it = headers.begin(); it != headers.end(); ++it )
        {
            const std::string key = it.key();
            const std::string value = headers.get<std::string>( key );
            req.set(key, value);
        }
    }

    void do_read() override {
        derived().ws().async_read(
            m_buffer,
            [ this ]( beast::error_code ec, std::size_t bytes_transfered )
            { handle_message( ec, bytes_transfered ); }
        );
    }

    void stop() override 
    {
        derived().ws().async_close( websocket::close_code::normal, [ this ]( beast::error_code ec ) {
            if(ec) CSP_THROW(RuntimeException, ec.message());
            m_on_close();
        });
    }

    void send( const std::string& s ) override
    {
        net::post(
            derived().ws().get_executor(),
            [this, s]()
            { 
                m_queue.push_back(s); 
                if (m_queue.size() > 1) return;
                do_write(m_queue.front());
            }
        );
    }

    void do_write(const std::string& s) override
    {
        derived().ws().async_write(
            net::buffer(s),
            [this](beast::error_code ec, std::size_t bytes_transfered)
            {
                // add logging here?
                m_queue.erase(m_queue.begin());
                boost::ignore_unused(bytes_transfered);
                if(ec) m_on_send_fail(ec.message());
                if(m_queue.size() >0) do_write(m_queue.front());
            }
        );
    }


public:
    tcp::resolver m_resolver;
    Dictionary* m_properties;
    void_cb m_on_open;
    string_cb m_on_fail;
    char_cb m_on_message;
    void_cb m_on_close;
    string_cb m_on_send_fail;

    beast::flat_buffer m_buffer;
    std::vector<std::string> m_queue;
};

class WebsocketSessionNoTLS final: public WebsocketSession<WebsocketSessionNoTLS> {
public:
    WebsocketSessionNoTLS(
        net::io_context& ioc, 
        Dictionary* properties,
        void_cb& on_open,
        string_cb& on_fail,
        char_cb& on_message,
        void_cb& on_close,
        string_cb& on_send_fail
    ) : WebsocketSession(
        ioc,
        properties,
        on_open,
        on_fail,
        on_message,
        on_close,
        on_send_fail
        ),
        m_ws(net::make_strand(ioc))
    { }

    void run() override {
        m_resolver.async_resolve(
            m_properties->get<std::string>("host").c_str(),
            m_properties->get<std::string>("port").c_str(),
            [this]( beast::error_code ec, tcp::resolver::results_type results ) {
                if(ec) {
                    m_on_fail(ec.message());
                    return;
                }
                // Set the timeout for the operation
                beast::get_lowest_layer(m_ws).expires_after(std::chrono::seconds(5));

                // Make the connection on the IP address we get from a lookup
                beast::get_lowest_layer(m_ws).async_connect(
                    results,
                    [this]( beast::error_code ec, tcp::resolver::results_type::endpoint_type ep )
                    {
                        // Turn off the timeout on the tcp_stream, because
                        // the websocket stream has its own timeout system.
                        if(ec) {
                            m_on_fail(ec.message());
                            return;
                        }

                        beast::get_lowest_layer(m_ws).expires_never();

                        m_ws.set_option(
                            websocket::stream_base::timeout::suggested(
                                beast::role_type::client));

                        m_ws.set_option(websocket::stream_base::decorator(
                            [this](websocket::request_type& req)
                            {
                                set_headers(req);
                                req.set(http::field::user_agent, "CSP WebsocketEndpoint");
                            }
                        ));

                        std::string host_ = m_properties->get<std::string>("host") + ':' + std::to_string(ep.port());
                        m_ws.async_handshake(
                            host_,
                            m_properties->get<std::string>("route"),
                            [this]( beast::error_code ec ) {
                                if(ec) {
                                    m_on_fail(ec.message());
                                    return;
                                }
                                m_on_open();
                                m_ws.async_read(
                                    m_buffer,
                                    [ this ]( beast::error_code ec, std::size_t bytes_transfered )
                                    { handle_message( ec, bytes_transfered ); }
                                );
                            }
                        );
                    }
                );
            }
        );
    }

    websocket::stream<beast::tcp_stream>& ws()
    { return m_ws; }
private:
    websocket::stream<beast::tcp_stream> m_ws;
};

class WebsocketSessionTLS final: public WebsocketSession<WebsocketSessionTLS> {
public:
    WebsocketSessionTLS(
        net::io_context& ioc, 
        ssl::context& ctx,
        Dictionary* properties,
        void_cb& on_open,
        string_cb& on_fail,
        char_cb& on_message,
        void_cb& on_close,
        string_cb& on_send_fail
    ) : WebsocketSession(
        ioc,
        properties,
        on_open,
        on_fail,
        on_message,
        on_close,
        on_send_fail
        ),
        m_ws(net::make_strand(ioc), ctx)
    { }

    void run() override {
        m_resolver.async_resolve(
            m_properties->get<std::string>("host").c_str(),
            m_properties->get<std::string>("port").c_str(),
            [this]( beast::error_code ec, tcp::resolver::results_type results ) {
                if(ec) {
                    m_on_fail(ec.message());
                    return;
                }
                // Set the timeout for the operation
                beast::get_lowest_layer(m_ws).expires_after(std::chrono::seconds(5));

                // Make the connection on the IP address we get from a lookup
                beast::get_lowest_layer(m_ws).async_connect(
                    results,
                    [this]( beast::error_code ec, tcp::resolver::results_type::endpoint_type ep )
                    {
                        if(ec) {
                            m_on_fail(ec.message());
                            return;
                        }

                        if(! SSL_set_tlsext_host_name(
                                m_ws.next_layer().native_handle(),
                                m_properties->get<std::string>("host").c_str()))
                        {
                            ec = beast::error_code(static_cast<int>(::ERR_get_error()),
                                net::error::get_ssl_category());
                            m_on_fail(ec.message());
                            return;
                        }

                        m_complete_host = m_properties->get<std::string>("host") + ':' + std::to_string(ep.port());

                        // ssl handler
                        m_ws.next_layer().async_handshake(
                            ssl::stream_base::client,
                            [this]( beast::error_code ec ) {
                                if(ec) {
                                    m_on_fail(ec.message());
                                    return;
                                }

                                beast::get_lowest_layer(m_ws).expires_never();
                                // Set suggested timeout settings for the websocket
                                m_ws.set_option(websocket::stream_base::timeout::suggested(beast::role_type::client));

                                // Set a decorator to change the User-Agent of the handshake
                                m_ws.set_option(websocket::stream_base::decorator(
                                    [this](websocket::request_type& req)
                                    {
                                        set_headers(req);
                                        req.set(http::field::user_agent, "CSP WebsocketAdapter");
                                    }));
                                
                                m_ws.async_handshake(
                                    m_complete_host,
                                    m_properties->get<std::string>("route"),
                                    [this]( beast::error_code ec ) {
                                        if(ec) {
                                            m_on_fail(ec.message());
                                            return;
                                        }
                                        m_on_open();
                                        m_ws.async_read(
                                            m_buffer,
                                            [ this ]( beast::error_code ec, std::size_t bytes_transfered )
                                            { handle_message( ec, bytes_transfered ); }
                                        );
                                    }
                                );

                            }
                        );
                    }
                );
            }
        );
    }

    websocket::stream<beast::ssl_stream<beast::tcp_stream>>& ws()
    { return m_ws; }

private:
    websocket::stream<beast::ssl_stream<beast::tcp_stream>> m_ws;
    std::string m_complete_host;
};

class WebsocketEndpoint {
public:
    WebsocketEndpoint( Dictionary properties );
    virtual ~WebsocketEndpoint() { };

    void setOnOpen(void_cb on_open);
    void setOnFail(string_cb on_fail);
    void setOnMessage(char_cb on_message);
    void setOnClose(void_cb on_close);
    void setOnSendFail(string_cb on_send_fail);
    Dictionary& getProperties();
    void run();
    void stop();
    void send(const std::string& s);

private:
    Dictionary m_properties;
    BaseWebsocketSession* m_session;
    net::io_context m_ioc;
    void_cb m_on_open;
    string_cb m_on_fail;
    char_cb m_on_message;
    void_cb m_on_close;
    string_cb m_on_send_fail;
};


}

#endif