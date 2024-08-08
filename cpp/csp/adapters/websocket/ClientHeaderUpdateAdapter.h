#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_HEADERUPDATEADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_HEADERUPDATEADAPTER_H

#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>
#include <csp/adapters/websocket/csp_autogen/websocket_types.h>

namespace csp::adapters::websocket
{
using namespace csp::autogen;

class ClientHeaderUpdateOutputAdapter final: public OutputAdapter
{
public:
    ClientHeaderUpdateOutputAdapter(
        Engine * engine,
        Dictionary& properties
    );

    void executeImpl() override;

    const char * name() const override { return "WebsocketClientHeaderUpdateAdapter"; }

private:
    Dictionary& m_properties;

};

}


#endif