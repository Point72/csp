#include <csp/engine/Consumer.h>
#include <csp/engine/EventPropagator.h>

namespace csp
{

EventPropagator::EventPropagator()
{
}

void EventPropagator::propagate()
{
    m_consumers.apply( []( Consumer * consumer, InputId id ) { consumer -> handleEvent( id ); } );
}


//ConsumerVector
EventPropagator::ConsumerVector::ConsumerVector()
{
    //if this is getting constructed that means we have > 1 consumer, so start with initialization to size 2
    m_capacity = 2;
    m_size = 0;
    m_consumers = ( ConsumerInfo * ) malloc( sizeof( ConsumerInfo ) * m_capacity );
    m_consumers = ( ConsumerInfo * ) ( uint64_t( m_consumers ) | 0x1 );
}

EventPropagator::ConsumerVector::~ConsumerVector()
{
    free( consumers() );
}

EventPropagator::ConsumerInfo * EventPropagator::ConsumerVector::findConsumer( Consumer * consumer, InputId id )
{
    ConsumerInfo * needle = consumers();
    ConsumerInfo * end = needle + m_size;

    ConsumerInfo match{ consumer, id };
    while( needle < end )
    {
        if( *needle == match )
            return needle;
        ++needle;
    }
    return nullptr;
}

void EventPropagator::ConsumerVector::push_back( Consumer * consumer, InputId id )
{
    if( m_size == m_capacity )
    {
        m_capacity *= 2;
        m_consumers = ( ConsumerInfo * ) realloc( consumers(), sizeof( ConsumerInfo ) * m_capacity );
        m_consumers = ( ConsumerInfo * ) ( uint64_t( m_consumers ) | 0x1 );
    }

    consumers()[m_size++] = ConsumerInfo( consumer, id );
}

bool EventPropagator::ConsumerVector::addConsumer( Consumer * consumer, InputId id, bool checkExists )
{
    bool exists = checkExists && ( findConsumer( consumer, id ) != nullptr );
    if( !exists )
        push_back( consumer, id );
    return !exists;
}

bool EventPropagator::ConsumerVector::removeConsumer( Consumer * consumer, InputId id )
{
    //called by make_active, same as addConsumer but ensures its not already in the list
    ConsumerInfo * ci = findConsumer( consumer, id );
    if( ci )
    {
        //avoid deleting mid-array, swap with last elem instead
        *ci = consumers()[ m_size - 1 ];
        --m_size;
        return true;
    }

    return false;
}

//Consumers union
EventPropagator::Consumers::Consumers()
{
    m_consumerData.flag = EMPTY;
}

EventPropagator::Consumers::~Consumers()
{
    if( !isEmpty() && !isSingle() )
        consumersVector().ConsumerVector::~ConsumerVector();
}

bool EventPropagator::Consumers::addConsumer( Consumer * consumer, InputId id, bool checkExists )
{
    if( isEmpty() )
    {
        single() = ConsumerInfo{ consumer, id };
    }
    else if( isSingle() )
    {
        if( single() == ConsumerInfo{ consumer, id } )
            return false;

        //Copy out the single info before the unionized vector in place new destroys it!
        ConsumerInfo singleCI{ single() };

        //transform from single to vector
        new( &consumersVector() ) ConsumerVector();
        consumersVector().addConsumer( singleCI.consumer, singleCI.inputId, false );
        consumersVector().addConsumer( consumer, id, false );
    }
    else
    {
        return consumersVector().addConsumer( consumer, id, checkExists );
    }

    return true;
}

bool EventPropagator::Consumers::removeConsumer( Consumer * consumer, InputId id )
{
    if( isEmpty() )
        return false;

    if( isSingle() )
    {
        if( single() == ConsumerInfo{ consumer, id } )
        {
            m_consumerData.flag = EMPTY;
            return true;
        }
        return false;
    }

    if( consumersVector().removeConsumer( consumer, id ) )
    {
        if( consumersVector().size() == 1 )
        {
            //Going from vector back to single
            //Copy over data before we whack the unionized memory space
            auto singleCI = consumersVector()[0];
            consumersVector().ConsumerVector::~ConsumerVector();
            single() = singleCI;
        }
        return true;
    }

    return false;
}

void EventPropagator::Consumers::clear()
{
    if( !isEmpty() )
    {
        if( isSingle() )
            m_consumerData.flag = EMPTY;
        else
            consumersVector().clear();
    }
}

}
