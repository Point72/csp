#ifndef _IN_CSP_ENGINE_SCHEDULER_H
#define _IN_CSP_ENGINE_SCHEDULER_H

#include <csp/core/BasicAllocator.h>
#include <csp/core/Time.h>
#include <cassert>
#include <functional>
#include <map>
#include <unordered_map>
#include <unordered_set>

namespace csp
{

class InputAdapter;

class Scheduler
{
    struct Event;

public:
    //For non-collapsing inputs, the callback should return a pointer to the input adapter that is
    //unrolling / not collapsing.  This will ensure pending events for a given input adapter will get deferred
    //to the next cycle until its completely unrolled.  This avoids a O(n^2) loop of reprocessing multiple pending events on
    //a given input adapter.  If the callback isnt for an input adapter or if it is able to consume / process the tick, return nullptr
    using Callback = std::function<const InputAdapter *()>;

    class Handle
    {
    public:
        Handle() { reset(); }
        Handle( const Handle & rhs ) = default;
        bool expired() const;
        bool active() const { return !expired(); }

        operator bool() const { return active(); }

        //Only valid if handle is active, otherwise returns NONE
        DateTime time() const;

        void reset()
        {
            event = nullptr;
            id = 0;
        }

        size_t hash() const { return id; }

    private:
        Handle( uint64_t id_, Event * event_ ) : id( id_ ), event( event_ )
        {}

        uint64_t id;
        Event * event;
        friend class Scheduler;
    };

    //This is used by DynamicNode when starting a new dynamic engine to keep track of all events scheduled for
    //"now" which should execute after start
    class DynamicEngineStartMonitor
    {
    public:
        DynamicEngineStartMonitor( Scheduler & s, DateTime now );
        //destructor executes new events added at "now"
        ~DynamicEngineStartMonitor();

    private:
        Scheduler & m_scheduler;
        Event     * m_lastEvent;
        DateTime    m_now;
    };

    Scheduler();
    ~Scheduler();

    //Callback can return false which means it should stick around
    //and re-execute next engine cycle
    Handle scheduleCallback( Handle reserved, DateTime time, Callback &&cb );
    Handle scheduleCallback( DateTime time, Callback &&cb );
    Handle rescheduleCallback( Handle handle, DateTime time );
    bool   cancelCallback( Handle handle );

    //useful to break circular dep of a callback needing its own handle
    Handle reserveHandle();

    bool hasEvents()           { return !m_map.empty() || m_pendingEvents.hasEvents(); }
    DateTime nextTime() const  { return m_pendingEvents.hasEvents() ? m_pendingEvents.time() : m_map.begin() -> first; }

    void executeNextEvents( DateTime now, Event * start = nullptr );

private:
    template<typename T>
    class MapAllocator
    {
    public:
        typedef T value_type;

        MapAllocator() : m_allocator( 32768, true, true )
        {}
        ~MapAllocator() {}

        T * allocate( std::size_t n )
        {
            assert( n == 1 );
            return m_allocator.allocate();
        }

        void deallocate( T* p, std::size_t n ) noexcept
        {
            assert( n == 1 );
            m_allocator.free( p );
        }

    private:
        TypedBasicAllocator<T> m_allocator;
    };

    struct EventList
    {
        Event * head;
        Event * tail;
    };

    using EventMap       = std::map<DateTime,EventList,std::less<DateTime>,MapAllocator<std::pair<const DateTime,EventList>>>;
    using EventAllocator = TypedBasicAllocator<Event>;

    struct Event
    {
        Event() : Event( nullptr, -1 ) {}
        Event( Callback && f, uint64_t id_ ) : next( nullptr ), prev( nullptr ),
                                               func( std::move( f ) ), id( id_ )
        {}

        Event             * next = nullptr;
        Event             * prev = nullptr;
        EventMap::iterator  mapIt;
        Callback            func;
        //NOTE THE IMPORTANCE of this not being the first member, we rely on setting this after dealloc
        //but BasicAllocator overrides the first 8 bytes for next -> link in the freelist
        uint64_t            id = -1;
    };

    //Pending events for non-collapsing inputs
    class PendingEvents
    {
    public:
        PendingEvents( Scheduler & scheduler );
        ~PendingEvents();

        PendingEvents( const PendingEvents & ) = delete;
        PendingEvents & operator=( const PendingEvents & rhs ) = delete;

        void addPendingEvent( const InputAdapter * adapter, Event * event, DateTime time );
        void cancelEvent( Event * event );
        void executeCycle();
        void clear();

        bool hasEvents() const { return !m_pendingEvents.empty(); }

        DateTime time() const { return m_time; }

    private:
        //NOTE the reason that this isnt simply kept as an unordered_map<InputAdapter*,Event> list is to ensure
        //simulated runs have complete reproducibility and are deterministic.  If we key by InputAdapter * pointer, ordering of
        //invocations will vary across runs ( this generally wouldnt matter if all code was side-effect free, but even so basket input
        //tickeditems() order depends on order of basket elemnt ticks in a given cycle )
        //Instead we maintain an unordered_map<InputAdapter*, list iterator> to ensure we keep an adapter once in the m_pendingEvents list
        //The list itself is the one used for iteration and execution of events, which should alwys be deterministic

        //EventList for a single input adapter
        struct PendingEventList
        {
            const InputAdapter * adapter;

            //sentinels
            Event head;
            Event tail;
        };

        //time of all pending events
        DateTime m_time;

        //Every list entry is *PER ADAPTER*, each entry will have a linked list of Event objects
        std::list<PendingEventList> m_pendingEvents;

        std::unordered_map<const InputAdapter *, std::list<PendingEventList>::iterator> m_pendingAdapters;
        Scheduler & m_scheduler;
    };

    friend class PendingEvents;

    EventMap       m_map;
    PendingEvents  m_pendingEvents;
    EventAllocator m_eventAllocator;
    uint64_t       m_unique;
};

inline bool Scheduler::Handle::expired() const
{
    return !event || event -> id != id;
}

inline DateTime Scheduler::Handle::time() const
{
    return expired() ? DateTime::NONE() : event -> mapIt -> first;
}

inline Scheduler::Handle Scheduler::reserveHandle()
{
    return Handle{ ++m_unique, nullptr };
}

inline Scheduler::Handle Scheduler::scheduleCallback( DateTime time, Callback &&cb )
{
    return scheduleCallback( reserveHandle(), time, std::move( cb ) );
}

inline Scheduler::Handle Scheduler::scheduleCallback( Handle reserved, DateTime time, Callback &&cb )
{
    Event * event = m_eventAllocator.allocate();
    new( event ) Event{ std::move( cb ), reserved.id };

    auto rv = m_map.emplace( time, EventList{ event, event } );
    if( !rv.second )
    {
        rv.first -> second.tail -> next = event;
        event -> prev = rv.first -> second.tail;
        rv.first -> second.tail = event;
    }

    event -> mapIt = rv.first;
    return Handle{ event -> id, event };
}

inline Scheduler::Handle Scheduler::rescheduleCallback( Handle handle, DateTime time )
{
    if( handle.expired() )
        CSP_THROW( ValueError, "attempting to reschedule expired handle" );

    auto cb = std::move( handle.event -> func );
    cancelCallback( handle );

    return scheduleCallback( time, std::move( cb ) );
}

};

namespace std
{

template<>
struct hash<csp::Scheduler::Handle>
{
    size_t operator()( const csp::Scheduler::Handle & handle ) const
    {
        return handle.hash();
    }
};

}

#endif
