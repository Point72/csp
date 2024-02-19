#include <csp/adapters/websocket_client/WSClientOutputAdapter.h>
#include <iostream>

namespace csp::adapters::wsclient {

WSClientOutputAdapter::WSClientOutputAdapter(
    Engine * engine,
    client* client,
    websocketpp::connection_hdl* hdl,
    bool* active
) : OutputAdapter( engine ), m_client( client ), m_hdl( hdl ), m_active(active)
{

};

void WSClientOutputAdapter::executeImpl()
{
    const std::string & value = input() -> lastValueTyped<std::string>();
    websocketpp::lib::error_code ec;
    m_client->send(*m_hdl, value, websocketpp::frame::opcode::value::TEXT, ec);
};

}