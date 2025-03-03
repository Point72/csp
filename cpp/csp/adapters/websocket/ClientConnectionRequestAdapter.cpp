#include <csp/adapters/websocket/ClientConnectionRequestAdapter.h>

namespace csp::adapters::websocket {

ClientConnectionRequestAdapter::ClientConnectionRequestAdapter(
    Engine * engine,
    WebsocketEndpointManager * websocketManager,
    bool is_subscribe,
    size_t caller_id,
    boost::asio::strand<boost::asio::io_context::executor_type>& strand

) : OutputAdapter( engine ),  
    m_websocketManager( websocketManager ),
    m_strand( strand ),
    m_isSubscribe( is_subscribe ),
    m_callerId( caller_id ),
    m_checkPerformed( is_subscribe ? false : true )  // we only need to check for pruned input adapters
{}

void ClientConnectionRequestAdapter::executeImpl()
{
    // One-time check for pruned status
    if (unlikely(!m_checkPerformed)) {
        m_isPruned = m_websocketManager->adapterPruned(m_callerId);
        m_checkPerformed = true;
    }

    // Early return if pruned
    if (unlikely(m_isPruned))
        return;

    std::vector<Dictionary> properties_list;
    for (auto& request : input()->lastValueTyped<std::vector<InternalConnectionRequest::Ptr>>()) {
        if (!request->allFieldsSet())
            CSP_THROW(TypeError, "All fields must be set in InternalConnectionRequest");
            
        Dictionary dict;
        dict.update("host", request->host());
        dict.update("port", request->port());
        dict.update("route", request->route());
        dict.update("uri", request->uri());
        dict.update("use_ssl", request->use_ssl());
        dict.update("reconnect_interval", request->reconnect_interval());
        dict.update("persistent", request->persistent());
        
        dict.update("headers", request -> headers() );
        dict.update("on_connect_payload", request->on_connect_payload());
        dict.update("action", request->action());
        dict.update("dynamic", request->dynamic());
        dict.update("binary", request->binary());
        
        properties_list.push_back(std::move(dict));
    }

    // We intentionally post here, we want the thread running
    // the strand to handle the connection request. We want to keep
    // all updates to internal data structures at graph run-time
    // to that thread.
    boost::asio::post(m_strand, [this, properties_list=std::move(properties_list)]() {
        for(const auto& conn_req: properties_list) {
            m_websocketManager->handleConnectionRequest(conn_req, m_callerId, m_isSubscribe);
        }
    });
};

}