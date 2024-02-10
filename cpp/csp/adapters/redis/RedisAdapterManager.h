#ifndef _IN_CSP_ADAPTERS_REDIS_REDISADAPTERMANAGER_H
#define _IN_CSP_ADAPTERS_REDIS_REDISADAPTERMANAGER_H

#include <csp/core/Enum.h>
#include <csp/core/Hash.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PushInputAdapter.h>

#include <sw/redis++/redis++.h>

#include <string>
#include <thread>
#include <unordered_map>
#include <vector>


namespace csp::adapters::kafka
{

class RedisConsumer;
class RedisPublisher;
class RedisSubscriber;

struct RedisStatusMessageTypeTraits
{
    enum _enum : unsigned char
    {
        OK = 0,
        MSG_DELIVERY_FAILED = 1,
        MSG_SEND_ERROR = 2,
        MSG_RECV_ERROR = 3,
        GENERIC_ERROR = 4,

        NUM_TYPES
    };

protected:
    _enum m_value;
};

using RedisStatusMessageType = csp::Enum<RedisStatusMessageTypeTraits>;
using RedisSubscriber = sw::redis::Subscriber;
using RedisPublisher = sw::redis::Redis;


//Top level AdapterManager object for all redis adapters in the engine
class RedisAdapterManager final : public csp::AdapterManager
{
public:
    RedisAdapterManager( csp::Engine * engine, const Dictionary & properties );
    ~RedisAdapterManager();

    const char * name() const override { return "RedisAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;
    void stop() override;

    DateTime processNextSimTimeSlice(DateTime time) override;

    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );
    OutputAdapter * getOutputAdapter( CspTypePtr & type, const Dictionary & properties );

    std::shared_ptr<RedisPublisher> getPPublisher(const std::string& pattern, const Dictionary & properties);
    std::shared_ptr<RedisPublisher> getPublisher(const std::string& key, const Dictionary & properties);

    std::shared_ptr<RedisSubscriber> getPSubscriber(const std::string& pattern, const Dictionary & properties);
    std::shared_ptr<RedisSubscriber> getSubscriber(const std::string& key, const Dictionary & properties);

    void forceShutdown( const std::string & err );

private:

    std::vector<std::shared_ptr<RedisPublisher>> m_ppublisherVector;
    std::vector<std::shared_ptr<RedisPublisher>> m_publisherVector;
    std::vector<std::shared_ptr<RedisSubscriber>> m_psubscriberVector;
    std::vector<std::shared_ptr<RedisSubscriber>> m_subscriberVector;

    std::unordered_map<std::string, std::shared_ptr<RedisPublisher>> m_ppublisherMap;
    std::unordered_map<std::string, std::shared_ptr<RedisPublisher>> m_publisherMap;
    std::unordered_map<std::string, std::shared_ptr<RedisSubscriber>> m_psubscriberMap;
    std::unordered_map<std::string, std::shared_ptr<RedisSubscriber>> m_subscriberMap;


    std::string m_host;
    int         m_port;
    std::string m_password;
    int         m_db;
    bool        m_keepAlive;
    int         m_connectTimeoutMs;
    int         m_socketTimeoutMs;
    int         m_resp;
    size_t      m_poolSize;
    int         m_poolWaitTimeoutMs;
    int         m_poolConnectionLifetimeMs;
    int         m_poolConnectionIdleTimeMs;

    // TODO TLS https://github.com/sewenew/redis-plus-plus?tab=readme-ov-file#tlsssl-support
    // TODO UNIX domain socket https://github.com/sewenew/redis-plus-plus?tab=readme-ov-file#api-reference
};

}

#endif
