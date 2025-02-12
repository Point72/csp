#ifndef _IN_CSP_ADAPTERS_KAFKA_KAFKACONSUMER_H
#define _IN_CSP_ADAPTERS_KAFKA_KAFKACONSUMER_H

#include <csp/adapters/kafka/KafkaAdapterManager.h>
#include <csp/adapters/kafka/KafkaSubscriber.h>
#include <csp/core/Hash.h>
#include <librdkafka/rdkafkacpp.h>
#include <string>
#include <thread>
#include <unordered_set>

namespace csp::adapters::kafka
{

class RebalanceCb;

class KafkaConsumer
{
public:
    KafkaConsumer( KafkaAdapterManager * mgr, const Dictionary & properties );
    ~KafkaConsumer();

    void addSubscriber( const std::string & topic, const std::string & key, KafkaSubscriber * subscriber );
    void poll();
    void start( DateTime starttime );
    void stop();

    void setNumPartitions( const std::string & topic, size_t num );

    void forceReplayCompleted();

private:

    struct TopicData
    {
        //Key -> Subscriber
        using SubscriberMap = std::unordered_map<std::string, std::vector<KafkaSubscriber*>>;
        SubscriberMap      subscribers;
        KafkaSubscriber *  wildcardSubscriber = nullptr;
        std::vector<bool>  partitionLive;
        bool               flaggedReplayComplete = false;

        void markReplayComplete()
        {
            if( !flaggedReplayComplete )
            {
                // Flag all regular subscribers
                for( auto& subscriberEntry : subscribers )
                {
                    for( auto* subscriber : subscriberEntry.second )
                        subscriber -> flagReplayComplete();
                }
                
                // Handle wildcard subscriber if present
                if( wildcardSubscriber )
                    wildcardSubscriber -> flagReplayComplete();
                
                flaggedReplayComplete = true;
            }
        }
    };

    std::unordered_map<std::string,TopicData> m_topics;
    KafkaAdapterManager *                     m_mgr;
    std::unique_ptr<RdKafka::KafkaConsumer>   m_consumer;
    std::unique_ptr<RebalanceCb>              m_rebalanceCb;
    std::unique_ptr<std::thread>              m_pollThread;
    volatile bool                             m_running;
};

}

#endif
