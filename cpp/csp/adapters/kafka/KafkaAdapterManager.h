#ifndef _IN_CSP_ADAPTERS_KAFKA_KAFKAADAPTERMANAGER_H
#define _IN_CSP_ADAPTERS_KAFKA_KAFKAADAPTERMANAGER_H

#include <csp/core/Enum.h>
#include <csp/core/Hash.h>
#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PushInputAdapter.h>
#include <librdkafka/rdkafkacpp.h>
#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace RdKafka
{

class Conf;
class DeliveryReportCb;
class EventCb;
class Producer;
}

namespace csp::adapters::kafka
{

class KafkaConsumer;
class KafkaPublisher;
class KafkaSubscriber;

struct KafkaStatusMessageTypeTraits
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

using KafkaStatusMessageType = csp::Enum<KafkaStatusMessageTypeTraits>;

//Top level AdapterManager object for all kafka adapters in the engine
class KafkaAdapterManager final : public csp::AdapterManager
{
public:
    KafkaAdapterManager( csp::Engine * engine, const Dictionary & properties );
    ~KafkaAdapterManager();

    const char * name() const override { return "KafkaAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;

    void stop() override;

    DateTime processNextSimTimeSlice( DateTime time ) override;

    //properties will have topic and fieldmap information, amongst other things
    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );
    OutputAdapter * getOutputAdapter( CspTypePtr & type, const Dictionary & properties );

    RdKafka::Conf * getConsumerConf() { return m_consumerConf.get(); }

    const Dictionary::Value & startOffsetProperty() const { return m_startOffsetProperty; }

    int pollTimeoutMs() const { return m_pollTimeoutMs; }

    void forceShutdown( const std::string & err );

    void markConsumerReplayDone( KafkaConsumer * consumer, const std::string & topic );
    void onMessage( RdKafka::Message * msg ) const;
    
private:

    using TopicKeyPair = std::pair<std::string, std::string>;

    KafkaConsumer * getConsumer( const Dictionary & properties );
    void setConfProperties( RdKafka::Conf * conf, const Dictionary & properties );
    void pollProducers();

    KafkaSubscriber * getSubscriber( const std::string & topic, const std::string & key, const Dictionary & properties );
    KafkaPublisher * getStaticPublisher( const TopicKeyPair & pair, const Dictionary & properties );
    KafkaPublisher * getDynamicPublisher( const std::string & topic, const Dictionary & properties );

    struct TopicData
    {
        //Key -> Subscriber
        using SubscriberMap = std::unordered_map<std::string, std::vector<KafkaSubscriber*>>;
        using ConsumerMap   = std::unordered_map<KafkaConsumer *, bool>;
        ConsumerMap        consumers;
        SubscriberMap      subscribers;
        KafkaSubscriber *  wildcardSubscriber = nullptr;
        std::atomic<bool>  flaggedReplayComplete = false;

        void addSubscriber( KafkaConsumer * consumer, const std::string & key, KafkaSubscriber * subscriber );        
        void markConsumerReplayDone( KafkaConsumer * consumer );
        void markReplayComplete();
    };
    
    using TopicMap = std::unordered_map<std::string,TopicData>;
    TopicMap                                   m_topics;
    
    using ConsumerVector = std::vector<std::shared_ptr<KafkaConsumer>>;
    ConsumerVector                             m_consumerVector;
    

    using StaticPublishers = std::unordered_map<TopicKeyPair, std::unique_ptr<KafkaPublisher>, hash::hash_pair>;
    StaticPublishers                           m_staticPublishers;

    using DynamicPublishers = std::vector<std::unique_ptr<KafkaPublisher>>;
    DynamicPublishers                          m_dynamicPublishers;

    using Subscribers = std::unordered_map<TopicKeyPair, std::unique_ptr<KafkaSubscriber>, hash::hash_pair>;
    Subscribers                                m_subscribers;

    int                                        m_pollTimeoutMs;
    size_t                                     m_maxThreads;
    size_t                                     m_consumerIdx;

    std::unique_ptr<RdKafka::EventCb>          m_eventCb;
    std::shared_ptr<RdKafka::Producer>         m_producer;
    std::unique_ptr<RdKafka::DeliveryReportCb> m_producerCb;
    std::unique_ptr<std::thread>               m_producerPollThread;
    std::atomic<bool>                          m_producerPollThreadActive;
    std::atomic<bool>                          m_unrecoverableError;

    std::unique_ptr<RdKafka::Conf>             m_consumerConf;
    std::unique_ptr<RdKafka::Conf>             m_producerConf;
    Dictionary::Value                          m_startOffsetProperty;
};

}

#endif
