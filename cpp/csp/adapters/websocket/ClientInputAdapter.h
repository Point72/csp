#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_INPUTADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_INPUTADAPTER_H

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

    void processMessage( void* c, size_t t, PushBatch* batch );

private:
    adapters::utils::MessageStructConverterPtr m_converter;

};

} 


#endif // _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_INPUTADAPTER_H