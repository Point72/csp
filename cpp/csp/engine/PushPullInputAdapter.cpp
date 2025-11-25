#include <csp/engine/PushPullInputAdapter.h>

namespace csp
{
PushPullInputAdapter::PushPullInputAdapter( Engine *engine, CspTypePtr &type, PushMode pushMode,
                                            PushGroup *group, bool adjustOutOfOrderTime )
            : PushInputAdapter(engine, type, pushMode, group),
              m_nextPullEvent(nullptr),
              m_tailEvent(nullptr),
              m_notifiedEndOfPull(false),
              m_adjustOutOfOrderTime(adjustOutOfOrderTime)
{
}

PushPullInputAdapter::~PushPullInputAdapter()
{
    //free up any unused events
    PushPullEvent * event = nextPullEvent();
    while( event )
    {
        delete event;
        event = nextPullEvent();
    }
}

void PushPullInputAdapter::start( DateTime start, DateTime end )
{
    auto * nextEvent = nextPullEvent();
    if( nextEvent )
        scheduleNextPullEvent( nextEvent );
}

void PushPullInputAdapter::stop()
{
    rootEngine() -> cancelCallback( m_timerHandle );
    //shouldnt need to lock at this point
    auto * replayCompleteEvent = new PushPullEvent( this, DateTime::NONE() );
    rootEngine() -> pushPullEventQueue().push( replayCompleteEvent );
}

void PushPullInputAdapter::scheduleNextPullEvent( PushPullEvent * nextEvent )
{
    //Note that we make nextEvent mutable in the lambda since we need to be able to update it in processNextPullEvent
    //which can return false to force a rescheduled re-attempt with a new event pointer
    m_timerHandle = rootEngine() -> scheduleCallback( nextEvent -> time,
                                                      [this, nextEvent]() mutable
                                                      {
                                                          return processNextPullEvent( nextEvent ) ? nullptr : this;
                                                      } );
}

bool PushPullInputAdapter::processNextPullEvent( PushPullEvent *& nextEvent )
{
    bool consumed = switchCspType( dataType(),
                                   [ this, &nextEvent ]( auto tag )
                                   {
                                       using T = typename decltype(tag)::type;
                                       TypedPushPullEvent<T> *tevent = static_cast<TypedPushPullEvent<T> *>( nextEvent );

                                       bool consumed = consumeTick( tevent -> data );
                                       assert( consumed );

                                       delete tevent;

                                       while( ( nextEvent = nextPullEvent() ) &&
                                              nextEvent -> time == rootEngine() -> now() )
                                       {
                                           tevent = static_cast<TypedPushPullEvent<T> *>( nextEvent );
                                           consumed = consumeTick( tevent -> data );
                                           if( !consumed )
                                               return false;
                                           delete tevent;
                                       }

                                       return true;
                                   } );

    if( consumed && nextEvent )
        scheduleNextPullEvent( nextEvent );

    return consumed;
}

PushPullEvent * PushPullInputAdapter::nextPullEvent()
{
    while( m_nextPullEvent == nullptr )
    {
        //Any PushPullInputAdapter instance can update events on any other adapter
        PushPullEvent * event = rootEngine() -> pushPullEventQueue().popAll();
        while( event )
        {
            PushPullEvent * next = event -> next;
            event -> adapter -> setNextPushPullEvent( event );
            event = next;
        }
    }

    //DateTime of None is the sentinel value for replay complete
    if( m_nextPullEvent -> time.isNone() )
        return nullptr;
    
    auto * event = m_nextPullEvent;
    m_nextPullEvent = m_nextPullEvent -> next;

    //Always force time before start to start.  There are two possibilities:
    //- User asked to replay from EARLIEST, so they should get all ticks replayed and we cant replay before starttime
    //- User asked to replay from STARTTIME in which case, if the adapter is written correctly, we shouldnt get ticks before starttime
    if( unlikely( event -> time < rootEngine() -> startTime() ) )
        event -> time = rootEngine() -> startTime();
    
    if( m_adjustOutOfOrderTime )
        event -> time = std::max( event -> time, rootEngine() -> now() );

    return event;    
}

}
