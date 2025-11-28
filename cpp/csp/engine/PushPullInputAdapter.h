#ifndef _IN_CSP_ENGINE_PUSHPULLINPUTADAPTER_H
#define _IN_CSP_ENGINE_PUSHPULLINPUTADAPTER_H

#include <csp/engine/PushInputAdapter.h>
#include <csp/engine/PushPullEvent.h>

namespace csp
{

//A variation of PushInputAdapter that lets you schedule historical data as well.  Used for adapters
//that can replay history and switch to realtime seamlessly ( ie kafka )
class PushPullInputAdapter : public PushInputAdapter
{
public:
    PushPullInputAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode,
                          PushGroup * group = nullptr, bool adjustOutOfOrderTime = false );
    ~PushPullInputAdapter();
    
    template<typename T>
    void pushTick( bool live, DateTime time, T &&value, PushBatch *batch = nullptr );

    void flagReplayComplete();

    void start( DateTime start, DateTime end ) override;
    void stop() override;

protected:

    virtual PushPullEvent * nextPullEvent();

    bool flaggedLive() const { return m_notifiedEndOfPull; }

    void setNextPushPullEvent( PushPullEvent * event )
    {
        if( !m_nextPullEvent )
            m_nextPullEvent = event;
        else
        {
            assert( m_tailEvent );
            assert( m_nextPullEvent );
            m_tailEvent -> next = event;
        }

        m_tailEvent = event;
        event -> next = nullptr;
    }
    
private:
    bool processNextPullEvent( PushPullEvent *& nextEvent );
    void scheduleNextPullEvent( PushPullEvent * nextEvent );
    
    Scheduler::Handle m_timerHandle;
    PushPullEvent   * m_nextPullEvent;
    PushPullEvent   * m_tailEvent;
    bool              m_notifiedEndOfPull; //flagged when we're done pushing pull values
    bool              m_adjustOutOfOrderTime;
};

inline void PushPullInputAdapter::flagReplayComplete()
{
    if( unlikely( !m_notifiedEndOfPull ) )
    {
        m_notifiedEndOfPull = true;
        auto * replayCompleteEvent = new PushPullEvent( this, DateTime::NONE() );
        rootEngine() -> pushPullEventQueue().push( replayCompleteEvent );
    }
}

template<typename T>
inline void PushPullInputAdapter::pushTick( bool live, DateTime time, T &&value, PushBatch *batch )
{
    static_assert( std::is_trivially_move_constructible<typename std::remove_reference<T>::type>::value ||
                   std::is_rvalue_reference<decltype( value )>::value, "Push tick value must be rvalue or native type" );

    if( live )
    {
        flagReplayComplete();
        PushInputAdapter::pushTick( std::forward<T>( value ), batch );
    }
    else
    {
        if( unlikely( m_notifiedEndOfPull ) )
            CSP_THROW( RuntimeException, "PushPullInputAdapter tried to push a sim tick after live tick" );

        PushPullEvent * event = new TypedPushPullEvent<T>( this, time, std::forward<T>(value) );
        rootEngine() -> pushPullEventQueue().push( event );
    }
}

}

#endif
