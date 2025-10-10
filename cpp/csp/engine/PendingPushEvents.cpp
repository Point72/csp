#include <csp/engine/PendingPushEvents.h>
#include <csp/engine/PushInputAdapter.h>

namespace csp
{

PendingPushEvents::PendingPushEvents()
{

}

PendingPushEvents::~PendingPushEvents()
{
    for( auto & entry : m_ungroupedEvents )
    {
        PushEvent * e = entry.second.head;
        while( e )
        {
            PushEvent * next = e -> next;
            deleteEvent( e );
            e = next;
        }
    }

    for( auto & entry : m_groupedEvents )
    {
        PushEvent * e = entry.second.head;
        while( e )
        {
            PushEvent * next = e -> next;
            deleteEvent( e );
            e = next;
        }
    }
}

void PendingPushEvents::deleteEvent( PushEvent * event )
{
    switchCspType( event -> adapter() -> dataType(),
                   [ event ]( auto tag ) {
                       using T = typename decltype(tag)::type;
                       delete static_cast<TypedPushEvent<T> *>( event );
                   } );
}

void PendingPushEvents::addPendingEvent( PushEvent * event )
{
    PushInputAdapter * adapter = event -> adapter();
    event -> next = nullptr;

    if( adapter -> group() )
    {
        auto rv = m_groupedEvents.emplace( adapter -> group(), EventList{ event, event } );
        if( !rv.second )
        {
            rv.first -> second.tail -> next = event;
            rv.first -> second.tail = event;
        }
    }
    else
    {
        assert( adapter -> pushMode() == PushMode::NON_COLLAPSING );

        auto rv = m_ungroupedEvents.emplace( adapter, EventList{ event, event } );
        if( !rv.second )
        {
            rv.first -> second.tail -> next = event;
            rv.first -> second.tail = event;
        }
    }
}

void PendingPushEvents::processPendingEvents( std::vector<PushGroup*> & dirtyGroups )
{
    for( auto it = m_ungroupedEvents.begin(); it != m_ungroupedEvents.end(); )
    {
        //for ungrouped events we'll never process more than one at a time
        auto * event = it -> second.head;
        it -> second.head = event -> next;
        bool rv = it -> first -> consumeEvent( event, dirtyGroups );
        (void)rv;
        assert( rv == true );

        if( it -> second.head == nullptr )
            it = m_ungroupedEvents.erase( it );
        else
            ++it;
    }

    for( auto it = m_groupedEvents.begin(); it != m_groupedEvents.end(); )
    {
        auto * group = it -> first;
        assert( group -> state == PushGroup::NONE );

        auto * event = it -> second.head;
        PushEvent * deferred_head = nullptr;
        PushEvent * deferred_tail = nullptr;
        while( event && group -> state != PushGroup::LOCKED )
        {
            PushEvent * next = event -> next;

            bool consumed = event -> adapter() -> consumeEvent( event, dirtyGroups );

            if( !consumed )
            {
                if( !deferred_head ) 
                {
                    deferred_head = event;
                    deferred_tail = event;
                }
                else
                {
                    deferred_tail -> next = event;
                    deferred_tail = event;
                }
            }

            event = next;
        }

        if( unlikely( deferred_head != nullptr ) )
        {
            deferred_tail -> flagGroupEnd(); //ensure we flag group end on the last deferred events in this batch
            deferred_tail -> next = event;   //ensure we link to the next batch
            event = deferred_head;           //ensure deferred list gets executes next cycle
        }

        if( event == nullptr )
            it = m_groupedEvents.erase( it );
        else
        {
            it -> second.head = event;
            ++it;
        }
    }
}

}
