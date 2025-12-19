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

    void poll();
    void start( DateTime starttime );
    void stop();

    void setPartitions(  std::vector<RdKafka::TopicPartition*> & partitions );
    void addTopic( const std::string & topic );

private:

    struct TopicData
    {
        struct PartitionInfo
        {
            bool receivedEOF = false;
            //For multi-consumer on a single topic not all partitions in the vector are valid
            bool valid       = false;
        };
        
        bool flaggedReplayComplete = false;
        std::vector<PartitionInfo> partitionInfo;
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
