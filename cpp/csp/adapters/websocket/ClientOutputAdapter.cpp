#include <csp/adapters/websocket/ClientOutputAdapter.h>

namespace csp::adapters::websocket {

ClientOutputAdapter::ClientOutputAdapter(
    Engine * engine,
    WebsocketEndpointManager * websocketManager,
    size_t caller_id,
    net::io_context& ioc,
    boost::asio::strand<boost::asio::io_context::executor_type>& strand
) : OutputAdapter( engine ), 
    m_websocketManager( websocketManager ),
    m_callerId( caller_id ),
    m_ioc( ioc ),
    m_strand( strand )
{ };

void ClientOutputAdapter::executeImpl()
{
    const std::string & value = input() -> lastValueTyped<std::string>();
    boost::asio::post(m_strand, [this, value=value]() {
        m_websocketManager->send(value, m_callerId);
    });
}

}