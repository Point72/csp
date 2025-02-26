#include <csp/core/Exception.h>
#include <csp/adapters/kafka/KafkaOutputAdapter.h>
#include <csp/adapters/kafka/KafkaPublisher.h>
#include <csp/adapters/utils/MessageWriter.h>
#include <csp/adapters/utils/JSONMessageWriter.h>

#include <librdkafka/rdkafkacpp.h>

namespace csp::adapters::kafka
{

KafkaPublisher::KafkaPublisher( KafkaAdapterManager * mgr, const Dictionary & properties, std::string topic ) : m_adapterMgr( *mgr ),
                                                                                                                m_engine( mgr -> engine() ),
                                                                                                                m_topic( std::move( topic ) )

{
    auto protocol = properties.get<std::string>( "protocol" );
    if( protocol == "JSON" )
        m_msgWriter = std::make_shared<utils::JSONMessageWriter>( properties );
    else if( protocol != "RAW_BYTES" )
        CSP_THROW( NotImplemented, "msg protocol " << protocol << " not currently supported for kafka output adapters" );
}

KafkaPublisher::~KafkaPublisher()
{
}

OutputAdapter * KafkaPublisher::getOutputAdapter( CspTypePtr & type, const Dictionary & properties, const std::string & key )
{
    if( isRawBytes() && !m_adapters.empty() )
        CSP_THROW( RuntimeException, "Attempting to publish multiple timeseries to kafka key " << key << " with RAW_BYTES protocol.  Only one output per key is allowed" );

    //create and register adapter
    auto adapter = m_engine -> createOwnedObject<KafkaOutputAdapter>( *this, type, properties, key );
    m_adapters.emplace_back( adapter );
    return m_adapters[ m_adapters.size() - 1 ];
}

OutputAdapter * KafkaPublisher::getOutputAdapter( CspTypePtr & type, const Dictionary & properties, const std::vector<std::string> & keyFields )
{
    if( isRawBytes() )
        CSP_THROW( RuntimeException, "vector of key fields is unsupported for RAW_BYTES protocol" );

    auto adapter = m_engine -> createOwnedObject<KafkaOutputAdapter>( *this, type, properties, keyFields );
    m_adapters.emplace_back( adapter );
    return m_adapters[ m_adapters.size() - 1 ];
}

PushInputAdapter * KafkaPublisher::getStatusAdapter()
{
    return nullptr;
}

void KafkaPublisher::start( std::shared_ptr<RdKafka::Producer> producer )
{
    m_producer = producer;

    std::unique_ptr<RdKafka::Conf> tconf( RdKafka::Conf::create( RdKafka::Conf::CONF_TOPIC ) );

    std::string errstr;
    m_kafkaTopic = std::shared_ptr<RdKafka::Topic>( RdKafka::Topic::create( m_producer.get(), m_topic, tconf.get(), errstr ) );
    if( !m_kafkaTopic )
        CSP_THROW( RuntimeException, "Failed to create RdKafka::Topic for producer on topic " << m_topic << ":" << errstr );
}

void KafkaPublisher::stop()
{
    //Note its important to release the topic here because its underlying mem lives in the kafka context of the Producer
    //if Producer is destroyed first, when we try to destroy the topic its a memory error
    m_kafkaTopic.reset();
}

void KafkaPublisher::send( const void * data, size_t len )
{
    RdKafka::ErrorCode err;
    err = m_producer -> produce( m_kafkaTopic.get(), RdKafka::Topic::PARTITION_UA, RdKafka::Producer::RK_MSG_COPY,
                                 const_cast<void *>( data ), len, m_key.c_str(), m_key.length(), nullptr );
    
    if( err != RdKafka::ERR_NO_ERROR )
    {
        std::string errMsg( "KafkaPublisher Error sending message " );
        errMsg += RdKafka::err2str( err );
        m_adapterMgr.pushStatus( StatusLevel::ERROR, KafkaStatusMessageType::MSG_SEND_ERROR, errMsg );
    }
}

void KafkaPublisher::onEndCycle()
{
    auto [data,len] = m_msgWriter -> finalize();
    send( data, len );
}

}
