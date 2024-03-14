#include <csp/adapters/websockets/ClientHeaderUpdateAdapter.h>

namespace csp::adapters::websockets {

ClientHeaderUpdateAdapter::ClientHeaderUpdateAdapter(
    Engine * engine,
    Dictionary& properties
) : OutputAdapter( engine ), m_properties(properties)
{

};

void ClientHeaderUpdateAdapter::executeImpl()
{
    std::vector<std::string> value = input() -> lastValueTyped<std::vector<std::string>>();
    DictionaryPtr headers = m_properties.get<DictionaryPtr>("headers");
    // this is hacky but it works...
    for( size_t i = 0; i < value.size(); i+=2)
    {
        const std::string key = value[i];
        const std::string v= value[i+1];
        headers->update(key, v);
    }
};

}