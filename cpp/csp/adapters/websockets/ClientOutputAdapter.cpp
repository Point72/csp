#include <csp/adapters/websockets/ClientOutputAdapter.h>

namespace csp::adapters::websockets {

ClientOutputAdapter::ClientOutputAdapter(
    Engine * engine,
    WebsocketEndpointBase * endpoint
) : OutputAdapter( engine ), m_endpoint(endpoint)
{ };

void ClientOutputAdapter::executeImpl()
{
    const std::string & value = input() -> lastValueTyped<std::string>();
    m_endpoint->send(value);
};

}