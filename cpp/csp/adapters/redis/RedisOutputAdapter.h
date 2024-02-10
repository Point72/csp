#ifndef _IN_CSP_ADAPTERS_REDIS_REDISOUTPUTADAPTER_H
#define _IN_CSP_ADAPTERS_REDIS_REDISOUTPUTADAPTER_H

#include <csp/adapters/utils/MessageWriter.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/OutputAdapter.h>
#include <string>

namespace csp::adapters::redis
{

class RedisPublisher;

class RedisOutputAdapter final: public OutputAdapter
{
public:
    RedisOutputAdapter( Engine * engine, RedisPublisher & publisher, CspTypePtr & type, const Dictionary & properties, const std::string & key );
    RedisOutputAdapter( Engine * engine, RedisPublisher & publisher, CspTypePtr & type, const Dictionary & properties, const std::vector<std::string> & keyFields );
    ~RedisOutputAdapter();

    void executeImpl() override;

    const char * name() const override { return "RedisOutputAdapter"; }

private:
    RedisOutputAdapter( Engine * engine, RedisPublisher & publisher, CspTypePtr & type, const Dictionary & properties );
    void addFields( const std::vector<std::string> & keyFields, CspTypePtr & type, size_t i = 0 );
    const std::string & getKey( const Struct * struct_ );

    RedisPublisher &            m_publisher;
    utils::OutputDataMapperPtr  m_dataMapper;
    std::vector<StructFieldPtr> m_structFields;
};

}

#endif
