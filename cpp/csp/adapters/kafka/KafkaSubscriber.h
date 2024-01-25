#ifndef _IN_CSP_ADAPTERS_KAFKA_KAFKASUBSCRIBER_H
#define _IN_CSP_ADAPTERS_KAFKA_KAFKASUBSCRIBER_H

#include <csp/adapters/kafka/KafkaAdapterManager.h>
#include <csp/adapters/kafka/KafkaInputAdapter.h>
#include <csp/engine/PushInputAdapter.h>

#include <librdkafka/rdkafkacpp.h>

namespace csp::adapters::kafka
{

class KafkaSubscriber
{
public:
    KafkaSubscriber( KafkaAdapterManager * mgr, const Dictionary & properties );
    ~KafkaSubscriber();

    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );

    void onMessage( RdKafka::Message* message, bool live );
    void flagReplayComplete();

private:
    using Adapters = std::vector<KafkaInputAdapter *>;
    Adapters              m_adapters;

    KafkaAdapterManager & m_adapterMgr;
    Engine *              m_engine;
    PushGroup             m_pushGroup;
};




}

#endif
