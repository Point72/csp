#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_ENDPOINT_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_ENDPOINT_H

// need a base -> TLS, base -> No TLS
#include <csp/adapters/websocket/ClientInputAdapter.h>
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/config/asio_no_tls_client.hpp>
#include <websocketpp/client.hpp>
#include <functional>

namespace csp::adapters::websocket {

using on_message_cb = std::function<void(std::string)>;
using on_send_fail_cb = std::function<void(const std::string&)>;
using void_cb = std::function<void()>;

using message_ptr = websocketpp::config::core_client::message_type::ptr;


class WebsocketEndpointBase {
public:
    WebsocketEndpointBase(csp::Dictionary properties);
    virtual ~WebsocketEndpointBase() { };

    virtual void run() { };
    virtual void send(const std::string&){ };
    virtual void close() { };

    void setOnMessageCb(on_message_cb cb);
    void setOnOpenCb(void_cb cb);
    void setOnFailCb(void_cb cb);
    void setOnCloseCb(void_cb cb);
    void setOnSendFailCb(on_send_fail_cb cb);

    csp::Dictionary& getProperties();

public:
    csp::Dictionary m_properties;
    void_cb m_on_open;
    on_message_cb m_on_message;
    void_cb m_on_fail;
    void_cb m_on_close;
    on_send_fail_cb m_on_send_fail;

};


class WebsocketEndpointTLS final: public WebsocketEndpointBase {

using tls_client = websocketpp::client<websocketpp::config::asio_tls_client>;

public:
    WebsocketEndpointTLS(const csp::Dictionary properties);
    ~WebsocketEndpointTLS();

    void run() override;
    void send(const std::string& s) override;
    void close() override;

private:
    tls_client m_client;
    websocketpp::connection_hdl m_hdl;
};


class WebsocketEndpointNoTLS final: public WebsocketEndpointBase {

using client = websocketpp::client<websocketpp::config::asio_client>;

public:
    WebsocketEndpointNoTLS(csp::Dictionary properties);
    ~WebsocketEndpointNoTLS();

    void run() override;
    void send(const std::string& s) override;
    void close() override;

private:
    client m_client;
    websocketpp::connection_hdl m_hdl;
};
}

#endif