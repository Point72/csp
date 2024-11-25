#include <csp/adapters/websocket/ClientHeaderUpdateAdapter.h>

namespace csp::adapters::websocket {

class WebsocketEndpointManager;

ClientHeaderUpdateOutputAdapter::ClientHeaderUpdateOutputAdapter(
    Engine * engine,
    WebsocketEndpointManager * mgr,
    boost::asio::strand<boost::asio::io_context::executor_type>& strand
) : OutputAdapter( engine ), m_mgr( mgr ), m_strand( strand )
{ };

void ClientHeaderUpdateOutputAdapter::executeImpl()
{
    Dictionary headers;
    for (auto& update : input()->lastValueTyped<std::vector<WebsocketHeaderUpdate::Ptr>>()) {
        if (update->key_isSet() && update->value_isSet()) {
            headers.update(update->key(), update->value());
        }
    }
    boost::asio::post(m_strand, [this, headers=std::move(headers)]() {
        auto endpoint = m_mgr -> getNonDynamicEndpoint();
        endpoint -> updateHeaders(std::move(headers));
    });
};

}