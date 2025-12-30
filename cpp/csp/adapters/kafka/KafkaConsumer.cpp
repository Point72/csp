#include <csp/adapters/kafka/KafkaConsumer.h>

#if 0
#define debug_printf( ... ) printf( __VA_ARGS__ )
#else
#define debug_printf( ... )
#endif

namespace csp::adapters::kafka
{

class RebalanceCb : public RdKafka::RebalanceCb
{
public:
    RebalanceCb( KafkaConsumer & consumer ) : m_consumer( consumer ),
                                              m_startOffset( RdKafka::Topic::OFFSET_INVALID )
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
            m_consumer.setPartitions( partitions );

            //Dont reset offsets if an offset was already set.  Subtle issue when running multiple Consumer threads
            //on the same topic.  Subsequent consumers as they come up and re-assign the partitions from the first
            //consumer.  We dont want to reset the offsets again since this will cause us to replay data that was already
            //processed.  Instead, query the current offsets and only apply offsets if they arent set yet
            RdKafka::ErrorCode rc = consumer -> committed( partitions, 10000 );
            if( rc )
                CSP_THROW( RuntimeException, "Failed to get kafka committed offsets: " << RdKafka::err2str( rc ) );

            for( auto * partition : partitions )
            {
                if( partition -> offset() == RdKafka::Topic::OFFSET_INVALID )
                {
                    if( !m_startTime.isNone() )
                    {
                        partition -> set_offset( m_startTime.asMilliseconds() );
                        std::vector<RdKafka::TopicPartition*> tmp{ partition };
                        rc = consumer -> offsetsForTimes( tmp, 10000 );
                        if( rc )
                            CSP_THROW( RuntimeException, "Failed to get kafka offsets for starttime " << m_startTime << ": " << RdKafka::err2str( rc ) );
                        debug_printf( "Set offset on topic %s consumer %p partition %d to %ld\n", partition -> topic().c_str(), consumer,
                                      partition -> partition(), partition -> offset() );
                    }
                    else if( m_startOffset != RdKafka::Topic::OFFSET_INVALID )
                        partition -> set_offset( m_startOffset );
                }
                else
                    debug_printf( "offset on topic %s consumer %p partition %d already set to %ld\n", partition -> topic().c_str(), consumer, partition -> partition(), partition -> offset() );
            }
            
            rc = consumer -> assign( partitions );
            
            if( rc )
                CSP_THROW( RuntimeException, "Failed to get kafka offsets for starttime " << m_startTime << ": " << RdKafka::err2str( rc ) );

            consumer -> commitSync( partitions );
        }
        else
        {
            //Since we run with auto-sync off, force commit offets here so rebalanced consumers see the latest
            consumer -> position( partitions );
            consumer -> commitSync( partitions );
            consumer -> unassign();
        }
    }

private:
    KafkaConsumer & m_consumer;
    DateTime        m_startTime;
    int64_t         m_startOffset;
};

KafkaConsumer::KafkaConsumer( KafkaAdapterManager * mgr, const Dictionary & properties ) : m_mgr( mgr ),
                                                                                           m_running( false )
{
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

void KafkaConsumer::start( DateTime starttime )
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
    
    std::vector<std::string> topics;
    for (const auto& [topic, _] : m_topics)
        topics.emplace_back( topic );

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

void KafkaConsumer::addTopic( const std::string & topic )
{
    m_topics.emplace( topic, TopicData{} );
}

void KafkaConsumer::setPartitions( std::vector<RdKafka::TopicPartition*> & partitions )
{
    //Clear out any previous assignments
    for( auto &[_,topicData] : m_topics )
        topicData.partitionInfo.clear();

    for( auto * partition : partitions )
    {
        auto & topicData = m_topics[ partition -> topic() ];
        int partitionIdx = partition -> partition();
        if( ( uint32_t ) partitionIdx >= topicData.partitionInfo.size() )
            topicData.partitionInfo.resize( partitionIdx + 1, {} );
        topicData.partitionInfo[ partitionIdx ].valid = true;
    }

    //Handle degenerate case where more threads than partitions were requested, we can get assigned 0 partitions and
    //should flag ourselves complete in this case
    for( auto &[topic,topicData] : m_topics )
    {
        debug_printf( "KafkaConsumer %p topic %s assigned %lu partitions\n", this, topic.c_str(), partitions.size() );
        if( topicData.partitionInfo.empty() )
        {
            m_mgr -> markConsumerReplayDone( this, topic );
            topicData.flaggedReplayComplete = true;
        }
    }
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

            //We tend to accumulate more cases over time of error states that leave the engine deadlocked on PushPull adapters.
            //This section is for cases where we get an error that is not topic specific, but is consumer specific, but we know its non-recoverable
            //if it gets too long, or we realize that ANY error here should stop the engine, we can just always make it stop
            if( unlikely( msg -> err() == RdKafka::ERR_GROUP_AUTHORIZATION_FAILED ) )
            {
                m_mgr -> forceShutdown( RdKafka::err2str( msg -> err() ) + " error: " + msg -> errstr() );
                continue;
            }

            if( msg -> err() == RdKafka::ERR_NO_ERROR && msg -> len() )
                m_mgr -> onMessage( msg.get() );
            //Not sure why, but it looks like we repeatedly get EOF callbacks even after the original one
            //may want to look into this.  Not an issue in practice, but seems like unnecessary overhead
            else if( msg -> err() == RdKafka::ERR__PARTITION_EOF )
            {
                auto const partition = msg -> partition();
                auto & topicData = m_topics[ msg -> topic_name() ];
                auto & partitionInfo = topicData.partitionInfo;
                if( ( uint32_t ) partition >= partitionInfo.size() )
                    partitionInfo.resize( partition + 1, {} );

                if( !partitionInfo[ partition ].receivedEOF )
                {
                    partitionInfo[ partition ].receivedEOF = true;
                    
                    debug_printf( "%p [DEBUG] %s EOF on %d\n", m_consumer.get(), DateTime::now().asString().c_str(), partition );
                    debug_printf( "Remaining: " );
                    for( size_t i = 0; i < partitionInfo.size(); ++i )
                    {
                        if( partitionInfo[i].valid && !partitionInfo[i].receivedEOF )
                            debug_printf( "%lu, ", i );
                    }
                    debug_printf("\n" );
                }

                //need this gaurd since we get this repeatedly after initial EOF
                if( !topicData.flaggedReplayComplete )
                {
                    bool allDone = true;
                    for( auto & info : partitionInfo )
                    {
                        if( info.valid && !info.receivedEOF )
                        {
                            allDone = false;
                            break;
                        }
                    }

                    if( allDone )
                    {
                        m_mgr -> markConsumerReplayDone( this, msg -> topic_name() );
                        topicData.flaggedReplayComplete = true;
                    }
                }
            }
            else
            {
                //In most cases we should not get here, if we do then something is wrong
                //safest bet is to release the pull adapter so it doesnt stall the engine and
                //we can let the error msg through
                m_mgr -> markConsumerReplayDone( this, msg -> topic_name() );

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
