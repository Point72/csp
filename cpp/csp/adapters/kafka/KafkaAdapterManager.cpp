#include <csp/adapters/kafka/KafkaAdapterManager.h>
#include <csp/adapters/kafka/KafkaConsumer.h>
#include <csp/adapters/kafka/KafkaPublisher.h>
#include <csp/adapters/kafka/KafkaSubscriber.h>
#include <csp/engine/Dictionary.h>
#include <csp/core/Platform.h>

#include <iostream>
#include <librdkafka/rdkafkacpp.h>

namespace csp
{

INIT_CSP_ENUM( csp::adapters::kafka::KafkaStatusMessageType,
               "OK",
               "MSG_DELIVERY_FAILED",
               "MSG_SEND_ERROR",
               "MSG_RECV_ERROR"
);

}

namespace csp::adapters::kafka
{

class DeliveryReportCb : public RdKafka::DeliveryReportCb
{
public:
    DeliveryReportCb( KafkaAdapterManager * mgr ) : m_adapterManager( mgr )
    {
    }

    void dr_cb( RdKafka::Message &message ) final
    {
        /* If message.err() is non-zero the message delivery failed permanently
         * for the message. */
        if( message.err() )
        {
            std::string msg = "KafkaPublisher: Message delivery failed for topic " + message.topic_name() + ". Failure: " + message.errstr();
            m_adapterManager -> pushStatus( StatusLevel::ERROR, KafkaStatusMessageType::MSG_DELIVERY_FAILED, msg );
        }
    }
private:
    KafkaAdapterManager * m_adapterManager;
};

class EventCb : public RdKafka::EventCb
{
public:
    EventCb( KafkaAdapterManager * mgr ) : m_adapterManager( mgr ) {}

    void event_cb( RdKafka::Event & event ) override
    {
        if( event.severity() < RdKafka::Event::EVENT_SEVERITY_NOTICE )
        {
            std::string errmsg = "KafkaConsumer: error ( " + std::to_string( event.err() ) + "): " + RdKafka::err2str( ( RdKafka::ErrorCode ) event.err() ) + ". Reason: " + event.str();
            m_adapterManager -> pushStatus( StatusLevel::ERROR, KafkaStatusMessageType::GENERIC_ERROR, errmsg );
        }

        if( event.type() == RdKafka::Event::EVENT_ERROR )
        {
            //We shutdown the app if its a fatal error OR if its an authentication issue which has plagued users multiple times
            //Adding ERR__ALL_BROKERS_DOWN which happens when all brokers are down
            if( event.fatal() ||
                event.err() == RdKafka::ErrorCode::ERR__AUTHENTICATION ||
                event.err() == RdKafka::ErrorCode::ERR__ALL_BROKERS_DOWN )
            {
                m_adapterManager -> forceShutdown( RdKafka::err2str( ( RdKafka::ErrorCode ) event.err() ) + event.str() );
            }
        }
    }

private:
    KafkaAdapterManager * m_adapterManager;
};

KafkaAdapterManager::KafkaAdapterManager( csp::Engine * engine, const Dictionary & properties ) : AdapterManager( engine ),
                                                                                                  m_consumerIdx( 0 ),
                                                                                                  m_producerPollThreadActive( false ),
                                                                                                  m_unrecoverableError( false )
{
    m_maxThreads = properties.get<uint64_t>( "max_threads" );
    m_pollTimeoutMs = properties.get<TimeDelta>( "poll_timeout" ).asMilliseconds();

    m_eventCb = std::make_unique<EventCb>( this );
    m_producerCb = std::make_unique<DeliveryReportCb>( this );

    std::string errstr;

    const Dictionary & rdKafkaProperties = *properties.get<DictionaryPtr>( "rd_kafka_conf_properties" );

    m_consumerConf.reset( RdKafka::Conf::create( RdKafka::Conf::CONF_GLOBAL ) );
    setConfProperties( m_consumerConf.get(), rdKafkaProperties );
    setConfProperties( m_consumerConf.get(), *properties.get<DictionaryPtr>( "rd_kafka_consumer_conf_properties" ) );
    if( properties.exists( "start_offset" ) )
    {
        //used later in start since we need starttime
        m_startOffsetProperty = properties.getUntypedValue( "start_offset" );
    }

    if( m_consumerConf -> set( "event_cb", m_eventCb.get(), errstr ) != RdKafka::Conf::CONF_OK )
        CSP_THROW( RuntimeException, "Failed to set consumer error cb: " << errstr );

    m_producerConf.reset( RdKafka::Conf::create( RdKafka::Conf::CONF_GLOBAL ) );
    setConfProperties( m_producerConf.get(), rdKafkaProperties );
    setConfProperties( m_producerConf.get(), *properties.get<DictionaryPtr>( "rd_kafka_producer_conf_properties" ) );
    if( m_producerConf -> set( "dr_cb", m_producerCb.get(), errstr ) != RdKafka::Conf::CONF_OK )
        CSP_THROW( RuntimeException, "Failed to set producer callback: " << errstr );
    if( m_producerConf -> set( "event_cb", m_eventCb.get(), errstr ) != RdKafka::Conf::CONF_OK )
        CSP_THROW( RuntimeException, "Failed to set producer error cb: " << errstr );
}

KafkaAdapterManager::~KafkaAdapterManager()
{
    // in case destructor is called before stop()
    if( m_producerPollThreadActive )
    {
        m_producerPollThreadActive = false;
        m_producerPollThread -> join();
    }
}

void KafkaAdapterManager::setConfProperties( RdKafka::Conf * conf, const Dictionary & properties )
{
    std::string errstr;

    for( auto it = properties.begin(); it != properties.end(); ++it )
    {
        std::string key = it.key();
        std::string value = properties.get<std::string>( key );
        if( conf -> set( key, value, errstr ) != RdKafka::Conf::CONF_OK )
            CSP_THROW( RuntimeException, "Failed to set property " << key << ": " << errstr );
    }
}

void KafkaAdapterManager::forceShutdown( const std::string & err )
{
    m_unrecoverableError = true;  // So we can alert the producer to stop trying to flush
    forceConsumerReplayComplete();
    try
    {
        CSP_THROW( RuntimeException, "Kafka fatal error. " +  err );
    }
    catch( const RuntimeException & )
    {
        rootEngine() -> shutdown( std::current_exception() );
    }
}

void KafkaAdapterManager::forceConsumerReplayComplete()
{
    for( auto & consumer : m_consumerVector )
        consumer -> forceReplayCompleted();
}

void KafkaAdapterManager::start( DateTime starttime, DateTime endtime )
{
    std::string errstr;

    if( !m_staticPublishers.empty() || !m_dynamicPublishers.empty() )
    {
        m_producer.reset( RdKafka::Producer::create( m_producerConf.get(), errstr ) );
        if ( !m_producer )
        {
            CSP_THROW( RuntimeException, "Failed to create producer: " << errstr );
        }
    }

    // start all consumers
    for( auto & it : m_consumerVector )
        it -> start( starttime );

    // start all publishers
    for( auto & it : m_staticPublishers )
        it.second -> start( m_producer );

    for( auto & it : m_dynamicPublishers )
        it -> start( m_producer );

    AdapterManager::start( starttime, endtime );

    if( !m_staticPublishers.empty() || !m_dynamicPublishers.empty() )
    {
        m_producerPollThreadActive = true;
        m_producerPollThread = std::make_unique<std::thread>( [ this ](){ pollProducers(); } );
    }
}

void KafkaAdapterManager::stop()
{
    AdapterManager::stop();

    // stop all consumers
    for( auto & it : m_consumerVector )
        it -> stop();

    if( m_producerPollThreadActive )
    {
        m_producerPollThreadActive = false;
        m_producerPollThread -> join();
    }

    // stop all publishers
    for( auto & it : m_staticPublishers )
        it.second -> stop();

    for( auto & it : m_dynamicPublishers )
        it -> stop();

    m_staticPublishers.clear();
    m_dynamicPublishers.clear();
    m_consumerVector.clear();
    m_producer.reset();
}

DateTime KafkaAdapterManager::processNextSimTimeSlice( DateTime time )
{
    // no sim data
    return DateTime::NONE();
}

void KafkaAdapterManager::pollProducers()
{
    while( m_producerPollThreadActive )
    {
        m_producer -> poll( m_pollTimeoutMs );
    }

    try
    {
        while( true )
        {
            auto rc = m_producer -> flush( 5000 );
            if( !rc || m_unrecoverableError )
                break;

            if( rc != RdKafka::ERR__TIMED_OUT )
                CSP_THROW( RuntimeException, "KafkaProducer failed to flush pending msgs on shutdown: " << RdKafka::err2str( rc ) );
        }
    }
    catch( ... )
    {
        rootEngine() -> shutdown( std::current_exception() );
    }
}

PushInputAdapter * KafkaAdapterManager::getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties )
{
    std::string topic = properties.get<std::string>( "topic" );
    std::string key = properties.get<std::string>( "key" );
    KafkaSubscriber * subscriber = this -> getSubscriber( topic, key, properties );
    return subscriber -> getInputAdapter( type, pushMode, properties );
}

OutputAdapter * KafkaAdapterManager::getOutputAdapter( CspTypePtr & type, const Dictionary & properties )
{
    std::string topic = properties.get<std::string>( "topic" );
    try
    {
        auto key = properties.get<std::string>( "key" );
        auto pair = TopicKeyPair( topic, key );
        KafkaPublisher * publisher = this -> getStaticPublisher( pair, properties );
        return publisher -> getOutputAdapter( type, properties, key );
    }
    catch( TypeError & e )
    {
        auto key = properties.get<std::vector<Dictionary::Data>>( "key" );
        std::vector<std::string> keyFields;
        for( auto & it : key )
            keyFields.emplace_back( std::get<std::string>( it._data ) );

        KafkaPublisher * publisher = this -> getDynamicPublisher( topic, properties );
        return publisher -> getOutputAdapter( type, properties, keyFields );
    }
}

KafkaConsumer * KafkaAdapterManager::getConsumer( const std::string & topic, const Dictionary & properties )
{
    // If we have seen this topic before, look up the consumer for it in the map
    // Otherwise, make a new consumer (and insert it into the map)
    // If we have reached m_maxThreads, then round-robin the topic onto a consumer (and insert it into the map)
    if( m_consumerMap.find( topic ) != m_consumerMap.end() )
    {
        return m_consumerMap[ topic ].get();
    }
    if( m_consumerVector.size() < m_maxThreads )
    {
        auto consumer = std::make_shared<KafkaConsumer>( this, properties );
        m_consumerVector.emplace_back( consumer );
        m_consumerMap.emplace( topic, consumer );
        return m_consumerMap[ topic ].get();
    }

    auto consumer = m_consumerVector[ m_consumerIdx++ ];
    m_consumerMap.emplace( topic, consumer );
    if( m_consumerIdx >= m_maxThreads )
        m_consumerIdx = 0;
    return consumer.get();
}

KafkaSubscriber * KafkaAdapterManager::getSubscriber( const std::string & topic, const std::string & key, const Dictionary & properties )
{
    auto pair = TopicKeyPair( topic, key );
    auto rv = m_subscribers.emplace( pair, nullptr );

    if( rv.second )
    {
        std::unique_ptr<KafkaSubscriber> subscriber( new KafkaSubscriber( this, properties ) );
        rv.first -> second = std::move( subscriber );

        this -> getConsumer( topic, properties ) -> addSubscriber( topic, key, rv.first -> second.get() );
    }

    return rv.first -> second.get();
}

// for static (string) keys, we create one publisher instance per <topic, key> pair
KafkaPublisher * KafkaAdapterManager::getStaticPublisher( const TopicKeyPair & pair, const Dictionary & properties )
{
    auto rv = m_staticPublishers.emplace( pair, nullptr );

    if( rv.second )
    {
        std::unique_ptr<KafkaPublisher> publisher( new KafkaPublisher( this, properties, pair.first ) );
        rv.first -> second = std::move( publisher );
    }

    KafkaPublisher * p = rv.first -> second.get();
    return p;
}

// for dynamic (struct) keys, we create one publisher instance per publish call
KafkaPublisher * KafkaAdapterManager::getDynamicPublisher( const std::string & topic, const Dictionary & properties )
{
    auto * p = new KafkaPublisher( this, properties, topic );
    m_dynamicPublishers.emplace_back( p );
    return p;
}

}
