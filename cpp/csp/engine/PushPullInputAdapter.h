#ifndef _IN_CSP_ENGINE_PUSHPULLINPUTADAPTER_H
#define _IN_CSP_ENGINE_PUSHPULLINPUTADAPTER_H

#include <csp/engine/PushInputAdapter.h>
#include <queue>

namespace csp
{
//A variation of PushInputAdapter that lets you schedule historical data as well.  Used for adapters
//that can replay history and switch to realtime seamlessly ( ie kafka )

class PushPullInputAdapter : public PushInputAdapter
{
public:
    PushPullInputAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode,
                          PushGroup * group = nullptr, bool adjustOutOfOrderTime = false );
    
    template<typename T>
    void pushTick( bool live, DateTime time, T &&value, PushBatch *batch = nullptr );

    void flagReplayComplete();

    void start( DateTime start, DateTime end ) override;
    void stop() override;

protected:
    
    struct PullDataEvent
    {
        DateTime time;
    };

    virtual PullDataEvent * nextPullEvent();

    bool flaggedLive() const { return m_notifiedEndOfPull; }

private:
    template<typename T>
    struct TypedPullDataEvent : public PullDataEvent
    {
        TypedPullDataEvent( DateTime t, T && d ) : PullDataEvent{ t },
                                                   data( std::forward<T>( d ) )
        {}

        typename std::remove_reference<T>::type data;
    };

    bool processNextPullEvent();

    using QueueT = std::queue<PullDataEvent *>;
    std::mutex        m_queueMutex;
    QueueT            m_threadQueue;
    QueueT            m_poppedPullEvents;
    Scheduler::Handle m_timerHandle;
    PullDataEvent   * m_nextPullEvent;
    bool              m_notifiedEndOfPull; //flagged when we're done pushing pull values
    bool              m_adjustOutOfOrderTime;

};

inline void PushPullInputAdapter::flagReplayComplete()
{
    if( unlikely( !m_notifiedEndOfPull ) )
    {
        m_notifiedEndOfPull = true;
        std::lock_guard<std::mutex> g( m_queueMutex );
        m_threadQueue.emplace( nullptr );
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

        //TBD allocators
        PullDataEvent * event = new TypedPullDataEvent<T>( time, std::forward<T>(value) );
        {
            std::lock_guard<std::mutex> g( m_queueMutex );
            m_threadQueue.emplace( event );
        }
    }
}

}

#endif
