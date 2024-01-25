#include <csp/adapters/kafka/KafkaInputAdapter.h>
#include <csp/adapters/kafka/KafkaConsumer.h>
#include <csp/adapters/kafka/KafkaSubscriber.h>
#include <csp/engine/Dictionary.h>

namespace csp::adapters::kafka
{

KafkaSubscriber::KafkaSubscriber( KafkaAdapterManager * mgr, const Dictionary & properties ) : m_adapterMgr( *mgr ),
                                                                                               m_engine( mgr -> engine() )
{
}

KafkaSubscriber::~KafkaSubscriber()
{
}

PushInputAdapter * KafkaSubscriber::getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties )
{
    //create and register adapter
    auto adapter = m_engine -> createOwnedObject<KafkaInputAdapter>( type, pushMode, &m_pushGroup, properties );

    m_adapters.emplace_back( adapter );
    return adapter;
}

void KafkaSubscriber::onMessage( RdKafka::Message* message, bool live )
{
    csp::PushBatch batch( m_engine -> rootEngine() );

    for( auto & adapter : m_adapters )
    {
        try
        {
            adapter -> processMessage( message, live, &batch );
        }
        catch( csp::Exception & err )
        {
            m_adapterMgr.pushStatus( StatusLevel::ERROR, KafkaStatusMessageType::MSG_RECV_ERROR, err.what() );
        }
    }
}

void KafkaSubscriber::flagReplayComplete()
{
    for( auto & adapter : m_adapters )
        adapter -> flagReplayComplete();
}

}
