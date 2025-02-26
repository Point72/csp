#include <csp/engine/PushPullInputAdapter.h>

namespace csp
{
PushPullInputAdapter::PushPullInputAdapter( Engine *engine, CspTypePtr &type, PushMode pushMode,
                                            PushGroup *group, bool adjustOutOfOrderTime )
            : PushInputAdapter(engine, type, pushMode, group),
              m_nextPullEvent(nullptr),
              m_notifiedEndOfPull(false),
              m_adjustOutOfOrderTime(adjustOutOfOrderTime)
{
    //free up any unused events
    while( m_nextPullEvent )
    {
        delete m_nextPullEvent;
        m_nextPullEvent = nextPullEvent();
    }
}

void PushPullInputAdapter::start( DateTime start, DateTime end )
{
    m_nextPullEvent = nextPullEvent();
    if( m_nextPullEvent )
    {
        m_timerHandle = rootEngine() -> scheduleCallback( m_nextPullEvent -> time,
                                                          [this]() { return processNextPullEvent() ? nullptr : this; } );
    }
}

void PushPullInputAdapter::stop()
{
    rootEngine() -> cancelCallback( m_timerHandle );
    //shouldnt need to lock at this point
    m_threadQueue.emplace( nullptr );
}

bool PushPullInputAdapter::processNextPullEvent()
{
    bool consumed = switchCspType( dataType(),
                                   [ this ]( auto tag )
                                   {
                                       using T = typename decltype(tag)::type;
                                       TypedPullDataEvent<T> *tevent = static_cast<TypedPullDataEvent<T> *>( m_nextPullEvent );

                                       bool consumed = consumeTick( tevent -> data );
                                       assert( consumed );

                                       delete tevent;

                                       while( ( m_nextPullEvent = nextPullEvent() ) &&
                                              m_nextPullEvent -> time == rootEngine() -> now() )
                                       {
                                           tevent = static_cast<TypedPullDataEvent<T> *>( m_nextPullEvent );
                                           consumed = consumeTick( tevent -> data );
                                           if( !consumed )
                                               return false;
                                           delete tevent;
                                       }

                                       return true;
                                   } );

    if( consumed && m_nextPullEvent )
    {
        m_timerHandle = rootEngine() -> scheduleCallback( m_nextPullEvent->time,
                                                          [this]() { return processNextPullEvent() ? nullptr : this; } );
    }

    return consumed;
}

PushPullInputAdapter::PullDataEvent * PushPullInputAdapter::nextPullEvent()
{
    //spin while we wait for data
    while( m_poppedPullEvents.empty() )
    {
        std::lock_guard<std::mutex> g( m_queueMutex );
        m_threadQueue.swap( m_poppedPullEvents );
    }

    auto * event = m_poppedPullEvents.front();
    m_poppedPullEvents.pop();

    if( event )
    {
        //Always force time before start to start.  There are two possibilities:
        //- User asked to replay from EARLIEST, so they should get all ticks replayed and we cant replay before starttime
        //- User asked to replay from STARTTIME in which case, if the adapter is written correctly, we shouldnt get ticks before starttime
        if( unlikely( event -> time < rootEngine() -> startTime() ) )
            event -> time = rootEngine() -> startTime();
        
        if( m_adjustOutOfOrderTime )
            event -> time = std::max( event -> time, rootEngine() -> now() );
    }

    return event;    
}

}
