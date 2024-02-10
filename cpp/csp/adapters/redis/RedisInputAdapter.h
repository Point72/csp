#ifndef _IN_CSP_ADAPTERS_REDIS_REDISINPUTADAPTER_H
#define _IN_CSP_ADAPTERS_REDIS_REDISINPUTADAPTER_H

#include <csp/adapters/utils/MessageStructConverter.h>
#include <csp/engine/PushPullInputAdapter.h>
#include <csp/engine/Struct.h>

namespace csp::adapters::redis
{

class RedisInputAdapter final: public PushPullInputAdapter
{
public:
    RedisInputAdapter( Engine * engine, CspTypePtr & type,
                       PushMode pushMode, PushGroup * group,
                       const Dictionary & properties );

    void processMessage(std::string message, bool live, csp::PushBatch* batch);

private:
    utils::MessageStructConverterPtr m_converter;
    StructFieldPtr m_partitionField;
    StructFieldPtr m_offsetField;
    StructFieldPtr m_liveField;
    StructFieldPtr m_timestampField;
    StructFieldPtr m_keyField;
};

}

#endif
