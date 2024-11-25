#include <csp/adapters/websocket/ClientConnectionRequestAdapter.h>
#include <csp/python/Conversions.h>
#include <Python.h>

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

    auto raw_val = input()->lastValueTyped<PyObject*>();
    auto val = python::fromPython<std::vector<Dictionary>>(raw_val);

    // We intentionally post here, we want the thread running
    // the strand to handle the connection request. We want to keep
    // all updates to internal data structures at graph run-time
    // to that thread.
    boost::asio::post(m_strand, [this, val=std::move(val)]() {
        for(const auto& conn_req: val) {
            m_websocketManager->handleConnectionRequest(conn_req, m_callerId, m_isSubscribe);
        }
    });
};

}