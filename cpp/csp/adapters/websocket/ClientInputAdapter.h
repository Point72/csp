#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_INPUTADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_INPUTADAPTER_H

#include <websocketpp/config/core_client.hpp>
#include <websocketpp/client.hpp>
#include <csp/engine/Dictionary.h>
#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/engine/Struct.h>

namespace csp::adapters::websocket
{


class ClientInputAdapter final: public PushInputAdapter {
public:
    ClientInputAdapter(
        Engine * engine,
        CspTypePtr & type,
        PushMode pushMode,
        const Dictionary & properties
    );

    void processMessage( std::string payload, PushBatch* batch );

private:
    adapters::utils::MessageStructConverterPtr m_converter;

};

} 


#endif // _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_INPUTADAPTER_H