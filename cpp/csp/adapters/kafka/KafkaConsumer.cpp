#include <csp/adapters/kafka/KafkaConsumer.h>

#include <iostream>

namespace csp::adapters::kafka
{

class RebalanceCb : public RdKafka::RebalanceCb
{
public:
    RebalanceCb( KafkaConsumer & consumer ) : m_consumer( consumer ),
                                              m_startOffset( RdKafka::Topic::OFFSET_INVALID ),
                                              m_doneSeeking( false )
    {
    }

    void setStartOffset( int64_t offset ) { m_startOffset = offset; }
    void setStartTime( DateTime time )    { m_startTime = time; }

    void rebalance_cb( RdKafka::KafkaConsumer *consumer,
                       RdKafka::ErrorCode err,
                       std::vector<RdKafka::TopicPartition*> & partitions ) override
    {
        if( err == RdKafka::ERR__ASSIGN_PARTITIONS )
        {
            if( !m_doneSeeking )
            {
                std::unordered_map<std::string,size_t> numPartitions;
                for( auto * partition : partitions )
                    numPartitions[ partition -> topic() ] += 1;

                for( auto & entry : numPartitions )
                    m_consumer.setNumPartitions( entry.first, entry.second );

                if( !m_startTime.isNone() )
                {
                    for( auto * partition : partitions )
                        partition -> set_offset( m_startTime.asMilliseconds() );

                    auto rc = consumer -> offsetsForTimes( partitions, 10000 );
                    if( rc )
                        CSP_THROW( RuntimeException, "Failed to get kafka offsets for starttime " << m_startTime << ": " << RdKafka::err2str( rc ) );
                }
                else
                {
                    for( auto * partition : partitions )
                        partition -> set_offset( m_startOffset );
                }

                auto rc = consumer -> assign( partitions );
                if( rc )
                    CSP_THROW( RuntimeException, "Failed to get kafka offsets for starttime " << m_startTime << ": " << RdKafka::err2str( rc ) );

                m_doneSeeking = true;
            }
            else
                consumer -> assign( partitions );
        }
        else
        {
            consumer -> unassign();
        }
    }

private:
    KafkaConsumer & m_consumer;
    DateTime        m_startTime;
    int64_t         m_startOffset;
    bool            m_doneSeeking;
};

KafkaConsumer::KafkaConsumer( KafkaAdapterManager * mgr, const Dictionary & properties ) : m_mgr( mgr ),
                                                                                           m_running( false )
{
    if( mgr -> startOffsetProperty().index() > 0 )
        m_rebalanceCb = std::make_unique<RebalanceCb>( *this );

    std::string errstr;
    auto * conf = m_mgr -> getConsumerConf();
    if( conf -> set( "rebalance_cb", m_rebalanceCb.get(), errstr ) != RdKafka::Conf::CONF_OK )
        CSP_THROW( RuntimeException, "Failed to set rebalance callback: " << errstr );

    m_consumer.reset( RdKafka::KafkaConsumer::create( conf, errstr ) );

    if( !m_consumer )
        CSP_THROW( RuntimeException, "Failed to create consumer: " << errstr );
}

KafkaConsumer::~KafkaConsumer()
{
    // in case destructor is called before stop()
    stop();
}

void KafkaConsumer::addSubscriber( const std::string & topic, const std::string & key, KafkaSubscriber * subscriber )
{
    if( key.empty() )
    {
        assert( m_topics[topic].wildcardSubscriber == nullptr );
        m_topics[topic].wildcardSubscriber = subscriber;
    }
    else
        m_topics[topic].subscribers[key].emplace_back( subscriber );
}

void KafkaConsumer::start( DateTime starttime )
{
    //RebalanceCb is only used / available if we requested a start_offset
    if( m_rebalanceCb )
    {
        auto & startOffsetProperty = m_mgr -> startOffsetProperty();
        if( std::holds_alternative<int64_t>( startOffsetProperty ) )
        {
            ReplayMode replayMode = ( ReplayMode ) std::get<int64_t>( startOffsetProperty );
            switch( replayMode )
            {
                case ReplayMode::EARLIEST:   m_rebalanceCb -> setStartOffset( RdKafka::Topic::OFFSET_BEGINNING ); break;
                case ReplayMode::LATEST:     m_rebalanceCb -> setStartOffset( RdKafka::Topic::OFFSET_END );       break;
                case ReplayMode::START_TIME: m_rebalanceCb -> setStartTime( starttime ); break;

                case ReplayMode::NUM_TYPES:                    
                case ReplayMode::UNKNOWN:
                    CSP_THROW( ValueError, "start_offset is unset" );
            }
        }
        else if( std::holds_alternative<DateTime>( startOffsetProperty ) )
        {
            auto dt = std::get<DateTime>( startOffsetProperty );
            m_rebalanceCb -> setStartTime( dt );
        }
        else if( std::holds_alternative<TimeDelta>( startOffsetProperty ) )
        {
            auto delta = std::get<TimeDelta>( startOffsetProperty );
            m_rebalanceCb -> setStartTime( starttime - delta.abs() );
        }
        else
            CSP_THROW( TypeError, "Expected enum, datetime or timedelta for startOffset" );
    }
    //This is a bit convoluted, but basically if we dont have rebalanceCB set, that means we are in "groupid" mode
    //which doesnt support seeking.  We force the adapters into a live mode, because groupid mode leads to deadlocks
    //on adapters that dont received any data since we dont have partition information available to declare them done ( we dont even connect to them all )
    else
        forceReplayCompleted();

    std::vector<std::string> topics;
    for (const auto& [topic, topic_data] : m_topics)
    {
        topics.emplace_back( topic );
        // wildcard subscription has no guarantee of being in order 
        // we flag replay complete as soon as we identify it.
        if( topic_data.wildcardSubscriber )
            topic_data.wildcardSubscriber -> flagReplayComplete();
    }

    RdKafka::ErrorCode err = m_consumer -> subscribe( topics );
    if( err )
        CSP_THROW( RuntimeException, "Failed to subscribe to " << m_topics.size() << " topics: " << RdKafka::err2str( err ) );

    m_running = true;
    m_pollThread = std::make_unique<std::thread>( [ this ](){ poll(); } );
}

void KafkaConsumer::stop()
{
    if( m_running )
    {
        m_running = false;
        m_pollThread -> join();
    }
    if( m_consumer.get() )
    {
        m_consumer -> close();
        m_consumer.reset();
    }
}

void KafkaConsumer::setNumPartitions( const std::string & topic, size_t num )
{
    auto & topicData = m_topics[ topic ];
    topicData.partitionLive.resize( num, false );
}

void KafkaConsumer::forceReplayCompleted()
{
    for( auto & entry : m_topics )
        entry.second.markReplayComplete();
}

void KafkaConsumer::poll()
{
    try
    {
        while( m_running )
        {
            std::unique_ptr<RdKafka::Message> msg( m_consumer -> consume( m_mgr -> pollTimeoutMs() ) );

            if( msg -> err() == RdKafka::ERR__TIMED_OUT )
                continue;

            auto topicIt = m_topics.find( msg -> topic_name() );
            if( topicIt == m_topics.end() )
            {
                std::string errmsg = "KafkaConsumer: Message received on unknown topic: " + msg -> topic_name() +
                    " errcode: " + RdKafka::err2str( msg -> err() ) + " error: " + msg -> errstr();
                m_mgr -> pushStatus( StatusLevel::ERROR, KafkaStatusMessageType::MSG_RECV_ERROR, errmsg );

                //We tend to accumulate more cases over time of error states that leave the engine deadlocked on PushPull adapters.
                //This section is for cases where we get an error that is not topic specific, but is consumer specific, but we know its non-recoverable
                //if it gets too long, or we realize that ANY error here should stop the engine, we can just always make it stop
                if( msg -> err() == RdKafka::ERR_GROUP_AUTHORIZATION_FAILED )
                    m_mgr -> forceShutdown( RdKafka::err2str( msg -> err() ) + " error: " + msg -> errstr() );
                continue;
            }

            auto & topicData = topicIt -> second;

            if( msg -> err() == RdKafka::ERR_NO_ERROR && msg -> len() )
            {
                if( !msg -> key() )
                {
                    std::string errmsg = "KafkaConsumer: Message received with null key on topic " + msg -> topic_name() + ".";
                    m_mgr -> pushStatus( StatusLevel::ERROR, KafkaStatusMessageType::MSG_RECV_ERROR, errmsg );
                    continue;
                }

                //printf( "Msg %s:%s on %d\n", msg -> topic_name().c_str(), msg -> key() -> c_str(), msg -> partition() );
                auto subscribersIt = topicData.subscribers.find( *msg -> key() );
                if( subscribersIt == topicData.subscribers.end() && topicData.wildcardSubscriber == nullptr )
                    continue;

                auto & partitionLive = topicData.partitionLive;
                if( ( uint32_t ) msg -> partition() >= partitionLive.size() )
                    partitionLive.resize( msg -> partition() + 1, false );

                bool live = partitionLive[ msg -> partition() ];
                if( subscribersIt != topicData.subscribers.end() )
                {
                    for( auto it : subscribersIt -> second )
                        it -> onMessage( msg.get(), live );
                }

                //Note we always have to tick wildcard as live because it can get messages from multiple
                //partitions, some which may have done replaying and some not ( not to mention that data can be out of order )
                if( topicData.wildcardSubscriber )
                    topicData.wildcardSubscriber -> onMessage( msg.get(), true );
            }

            //Not sure why, but it looks like we repeatedly get EOF callbacks even after the original one
            //may want to look into this.  Not an issue in practice, but seems like unnecessary overhead
            else if( msg -> err() == RdKafka::ERR__PARTITION_EOF )
            {
                auto const partition = msg -> partition();
                auto & partitionLive = topicData.partitionLive;
                if( ( uint32_t ) partition >= partitionLive.size() )
                    partitionLive.resize( partition + 1, false );

                partitionLive[ partition ] = true;

                //need this gaurd since we get this repeatedly after initial EOF
                if( !topicData.flaggedReplayComplete )
                {
                    bool allDone = true;
                    for( auto live : partitionLive )
                    {
                        if( !live )
                        {
                            allDone = false;
                            break;
                        }
                    }

                    //we need to flag end in case the topic doesnt have any incoming data, we cant stall the engine on the pull side of the adapter
                    if( allDone )
                        topicData.markReplayComplete();
                }
            }
            else
            {
                //In most cases we should not get here, if we do then something is wrong
                //safest bet is to release the pull adapter so it doesnt stall the engine and
                //we can let the error msg through
                topicData.markReplayComplete();

                std::string errmsg = "KafkaConsumer: Message error on topic \"" + msg -> topic_name() + "\". errcode: " + RdKafka::err2str( msg -> err() ) + " error: " + msg -> errstr();
                m_mgr -> pushStatus( StatusLevel::ERROR, KafkaStatusMessageType::MSG_RECV_ERROR, errmsg );
            }
        }
    }
    catch( const Exception & err )
    {
        m_mgr -> rootEngine() -> shutdown( std::current_exception() );
    }
}

}
