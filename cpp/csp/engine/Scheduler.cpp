#include <csp/engine/Scheduler.h>

namespace csp
{

#if 0
#define debug_printf( ... ) printf( __VA_ARGS__ )
#else
#define debug_printf( ... )
#endif

Scheduler::Scheduler() : m_pendingEvents( *this ), m_unique( 0 )
{
    //should make configurable at some point
    m_eventAllocator.init( 32768, false, true );
}

Scheduler::~Scheduler()
{
    for( auto it = m_map.begin(); it != m_map.end(); ++it )
    {
        Event * event = it -> second.head;
        while( event )
        {
            Event * nextEvent = event -> next;
            event -> ~Event();
            event = nextEvent;
        }
    }

    m_pendingEvents.clear();
}

//PendingEvents processing
Scheduler::PendingEvents::PendingEvents( Scheduler & scheduler ) : m_scheduler( scheduler )
{
}

Scheduler::PendingEvents::~PendingEvents()
{
}

void Scheduler::PendingEvents::clear()
{
    for( auto it = m_pendingEvents.begin(); it != m_pendingEvents.end(); it++ )
    {
        // skip past the sentinel, which is on the stack
        Event * event = it -> head.next;

        while( event != &it -> tail )
        {
            Event * nextEvent  = event -> next;
            event -> ~Event();
            event = nextEvent;
        }
    }
}

void Scheduler::PendingEvents::addPendingEvent( const InputAdapter * adapter, Event * event, DateTime now )
{
    //reset mapIt so event knows its no longer in main scheduler list ( needed for cancel handling )
    event -> mapIt = m_scheduler.m_map.end();

    //unconditionally overwrite, no need for conditional check here
    assert( m_pendingAdapters.empty() || m_time == now );
    m_time = now;

    debug_printf( "Adding pending event %p for adapter %p\n", event, adapter );

    auto rv = m_pendingAdapters.emplace( adapter, std::list<PendingEventList>::iterator() );
    if( rv.second )
    {
        auto peIt = m_pendingEvents.emplace( m_pendingEvents.end() );
        rv.first -> second = peIt;

        PendingEventList & el = *peIt;
        el.adapter = adapter;
        el.head.next = &el.tail;
        el.tail.prev = &el.head;

        debug_printf( "new entry for adapter %p\n", adapter );
    }

    auto & el = *rv.first -> second;

    el.tail.prev -> next = event;
    event -> prev = el.tail.prev;

    el.tail.prev = event;
    event -> next = &el.tail;
}

void Scheduler::PendingEvents::cancelEvent( Event * event )
{
    //we maintain sentinel head/tail values so that we can just erase in place without much thought
    event -> prev -> next = event -> next;
    event -> next -> prev = event -> prev;
}

void Scheduler::PendingEvents::executeCycle()
{
    debug_printf( "Scheduler::PendingEvents::executeCycle\n" );
    //execute one event per pending adapter
    for( auto elt = m_pendingEvents.begin(); elt != m_pendingEvents.end(); )
    {
        //It could be that we have an empty list due to a cancel of the last event
        Event * event = elt -> head.next;

        debug_printf( "Processing pending event %p on adapter %p\n", event, elt -> adapter );
        if( unlikely( event == &elt -> tail ) )
        {
            elt = m_pendingEvents.erase( elt );
            continue;
        }

        auto eventId = event -> id;
        event -> id = -1;

        if( unlikely( event -> func() != nullptr ) )
        {
            //Note its possible this can return false, even though we do one at a time
            //This is how PullInputAdapter is already implemented, it keeps consuming if next returns the same timestamp ( needed for bursts )
            //If that is ever reworked, we can add an exception for returning false here
            //CSP_THROW( RuntimeException, "Logical error process pending schedule events" );
            event -> id = eventId;
            ++elt;
            continue;
        }

        //list is done
        if( event -> next == &elt -> tail )
        {
            debug_printf( "end of list, removing adapter %p\n", elt -> adapter );
            m_pendingAdapters.erase( elt -> adapter );
            elt = m_pendingEvents.erase( elt );
        }
        else
        {
            debug_printf( "events remaining on adapter %p, setting head -> next to %p\n", elt -> adapter, event -> next );
            elt -> head.next = event -> next;
            event -> next -> prev = &elt -> head;
            ++elt;
        }

        event -> ~Event();
        m_scheduler.m_eventAllocator.free( event );
    }
}


//Regular non-pending handling
bool Scheduler::cancelCallback( Handle handle )
{
    if( handle.expired() )
        return false;

    Event * event = handle.event;

    //in pending events lists, no longer under m_maps domain
    if( event -> mapIt == m_map.end() )
    {
        m_pendingEvents.cancelEvent( event );
    }
    //now back to your regularly scheduled program
    else
    {
        //only event at this time slice
        if( event -> prev == nullptr && event -> next == nullptr )
        {
            m_map.erase( event -> mapIt );
        }
        else
        {
            EventList * list = &event -> mapIt -> second;
            if( event -> next )
                event -> next -> prev = event -> prev;
            else //else we are tail
                list -> tail = event -> prev;

            if( event -> prev )
                event -> prev -> next = event -> next;
            else //else we are head
                list -> head = event -> next;
        }
    }

    event -> ~Event();
    m_eventAllocator.free( event );
    event -> id = -1;
    return true;
}

void Scheduler::executeNextEvents( DateTime now, Event * start )
{
    debug_printf( "=== New cycle %s === \n", now.asString().c_str() );
    if( unlikely( m_pendingEvents.hasEvents() ) )
        m_pendingEvents.executeCycle();

    if( m_map.empty() || m_map.begin() -> first > now )
        return;

    EventList & elist = m_map.begin() -> second;

    // startPrev kept locally here since start wil be freed below
    Event * startPrev = unlikely( start != nullptr ) ? start -> prev : nullptr;
    // Note event is kept as a reference so that it gets updated throughout the loop
    Event *& event = unlikely( start != nullptr ) ? start : elist.head;
    // mark end before iterating in case timers with same time are added during loop
    Event * tail = elist.tail;

    while( true )
    {
        //invalidate event / handle before calling in case cb decides to reschedule itself
        auto eventId = event -> id;
        event -> id = -1;
        Event * cur = event;

        debug_printf( "Main loop processing event %p with id %lu\n", cur, eventId );
        auto * deferredAdapter = cur -> func();
        event = cur -> next;

        if( unlikely( deferredAdapter != nullptr ) )
        {
            //set id back to what it was, alarm is still cancellable
            cur -> id = eventId;
            m_pendingEvents.addPendingEvent( deferredAdapter, cur, now );
        }
        else
        {
            cur -> ~Event();
            m_eventAllocator.free( cur );
        }

        if( cur == tail )
            break;
    }

    //got events for same time during loop
    if( event )
    {
        //if we started from the middle of the list, link from where we started, otherwise we would miss these events
        //Note that start is only set when starting dynamic engines
        if( startPrev )
            event -> prev = startPrev;
    }
    else if( startPrev )
    {
        //if we have startPrev that means this is a dynamic engine start flush of events scheduled for "now" in engine
        //construction.  in this case we want to ensure the dynamic events are no longer in the queue
        startPrev -> next = nullptr;
        elist.tail = startPrev;
    }
    else
        m_map.erase( m_map.begin() );
}

Scheduler::DynamicEngineStartMonitor::DynamicEngineStartMonitor( Scheduler & s,
                                                                 DateTime now ) : m_scheduler( s ),
                                                                                  m_now( now )
{
    //note we only keep track of main map time, not pending events, since all initial dynamic events
    //will schedule into main
    DateTime nextMapTime = m_scheduler.m_map.begin() -> first;
    if( m_scheduler.hasEvents() && nextMapTime == now )
    {
        m_lastEvent = m_scheduler.m_map.begin() -> second.tail;
        assert( m_lastEvent -> next == nullptr );
    }
    else
        m_lastEvent = nullptr;
    debug_printf( "DynamicEngineStartMonitor::DynamicEngineStartMonitor() lastEvent %p\n", m_lastEvent );
}

Scheduler::DynamicEngineStartMonitor::~DynamicEngineStartMonitor()
{
    DateTime nextMapTime = m_scheduler.m_map.begin() -> first;
    if( m_scheduler.hasEvents() && nextMapTime == m_now &&
        ( !m_lastEvent ||                  //we had no events at "now" previously
          m_lastEvent -> next != nullptr ) // or there were events and we added new ones
        )
    {
        Event * startEvent = m_lastEvent ? m_lastEvent -> next : nullptr;

        debug_printf( "DynamicEngineStartMonitor::~DynamicEngineStartMonitor() startEvent %p\n", startEvent );
        m_scheduler.executeNextEvents( m_now, startEvent );
    }
    else
        debug_printf( "DynamicEngineStartMonitor::~DynamicEngineStartMonitor() no events\n" );

}

}
