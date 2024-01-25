#ifndef _IN_CSP_ADAPTERS_KAFKA_KAFKAPUBLISHER_H
#define _IN_CSP_ADAPTERS_KAFKA_KAFKAPUBLISHER_H

#include <csp/adapters/kafka/KafkaAdapterManager.h>
#include <string>
#include <thread>

namespace RdKafka
{

class DeliveryReportCb;
class Producer;
class Topic;

}

namespace csp::adapters::utils
{
class MessageWriter;
}

namespace csp::adapters::kafka
{

class KafkaOutputAdapter;

class KafkaPublisher : public EndCycleListener
{
public:
    KafkaPublisher( KafkaAdapterManager * mgr, const Dictionary & properties, std::string topic );
    ~KafkaPublisher();

    OutputAdapter * getOutputAdapter( CspTypePtr & type, const Dictionary & properties, const std::string & key );
    OutputAdapter * getOutputAdapter( CspTypePtr & type, const Dictionary & properties, const std::vector<std::string> & keyFields );

    PushInputAdapter * getStatusAdapter();

    void start( std::shared_ptr<RdKafka::Producer> producer );
    void stop();

    void setKey( std::string key ) { m_key = std::move( key ); }

    void onEndCycle() final;

    bool isRawBytes() const { return ( bool ) !m_msgWriter; }

    utils::MessageWriter & msgWriter() { return *m_msgWriter; }

    void scheduleEndCycleEvent()
    {
        m_adapterMgr.rootEngine() -> scheduleEndCycleListener( this );
    }

    void send( const void * data, size_t len );

private:

    using Adapters = std::vector<KafkaOutputAdapter*>;
    Adapters                              m_adapters;

    KafkaAdapterManager &                 m_adapterMgr;
    Engine *                              m_engine;
    std::shared_ptr<RdKafka::Producer>    m_producer;
    std::shared_ptr<utils::MessageWriter> m_msgWriter;
    std::shared_ptr<RdKafka::Topic>       m_kafkaTopic;
    std::string                           m_topic;
    std::string                           m_key;
};

}

#endif
