#ifndef _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_HEADERUPDATEADAPTER_H
#define _IN_CSP_ADAPTERS_WEBSOCKETS_CLIENT_HEADERUPDATEADAPTER_H

#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <csp/adapters/utils/MessageWriter.h>

namespace csp::adapters::websockets
{

class ClientHeaderUpdateAdapter final: public OutputAdapter
{
public:
    ClientHeaderUpdateAdapter(
        Engine * engine,
        Dictionary& properties
    );

    void executeImpl() override;

    const char * name() const override { return "ClientHeaderUpdateAdapter"; }

private:
    Dictionary& m_properties;

};

}


#endif