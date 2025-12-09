#ifndef _IN_CSP_ENGINE_ROOTENGINE_H
#define _IN_CSP_ENGINE_ROOTENGINE_H

#include <csp/core/Exception.h>
#include <csp/core/SRMWLockFreeQueue.h>
#include <csp/core/System.h>
#include <csp/core/Time.h>
#include <csp/engine/CycleStepTable.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/Engine.h>
#include <csp/engine/PendingPushEvents.h>
#include <csp/engine/Profiler.h>
#include <csp/engine/PushEvent.h>
#include <csp/engine/PushPullEvent.h>
#include <csp/engine/Scheduler.h>
#include <memory>

namespace csp
{

class Dictionary;

class EndCycleListener
{
public:
    virtual ~EndCycleListener() {};
    virtual void onEndCycle() = 0;
    
    bool isDirty() const  { return m_dirty; }
    void setDirtyFlag()   { m_dirty = true; }
    void clearDirtyFlag() { m_dirty = false; }

private:
    bool m_dirty = false;
};

class RootEngine : public Engine
{
    using PushEventQueue     = SRMWLockFreeQueue<PushEvent>;
    using PushPullEventQueue = SRMWLockFreeQueue<PushPullEvent>;

public:
    RootEngine( const Dictionary & );
    ~RootEngine();

    DateTime now() const { return m_now; }

    DateTime startTime() const  { return m_startTime; }
    DateTime endTime() const    { return m_endTime; }

    uint64_t cycleCount() const { return m_cycleCount; }

    DictionaryPtr engine_stats() const;
    csp::Profiler* profiler() const { return m_profiler.get(); }

    void     run( DateTime start, DateTime end );
    void     shutdown();
    void     shutdown( std::exception_ptr except );

    Scheduler::Handle reserveSchedulerHandle();
    Scheduler::Handle scheduleCallback( TimeDelta delta, Scheduler::Callback cb );
    Scheduler::Handle scheduleCallback( DateTime time, Scheduler::Callback cb );
    Scheduler::Handle scheduleCallback( Scheduler::Handle reservedHandle, DateTime time, Scheduler::Callback cb );
    Scheduler::Handle scheduleCallback( Scheduler::Handle reservedHandle, TimeDelta time, Scheduler::Callback cb );
    Scheduler::Handle rescheduleCallback( Scheduler::Handle handle, TimeDelta delta );
    Scheduler::Handle rescheduleCallback( Scheduler::Handle handle, DateTime time );

    void     cancelCallback( Scheduler::Handle handle );

    void     schedulePushEvent( PushEvent * event )             { m_pushEventQueue.push( event ); }
    void     schedulePushBatch( PushEventQueue::Batch & batch ) { m_pushEventQueue.push( batch ); }

    bool     scheduleEndCycleListener( EndCycleListener * l );

    //returns true if the engine is currently in runRealtime
    bool inRealtime() const         { return m_inRealtime; }

    //returns true if engine is configured realtime ( inRealtime can still be false if
    //realtime run starts in the past )
    bool configuredRealtime() const { return m_settings.realtime; }

    bool configuredProfile() const { return ( bool ) m_profiler; }

    //internally used by DynamicNode for dynamic engine creation
    Scheduler::DynamicEngineStartMonitor dynamicEngineStartMonitor()
    {
        return Scheduler::DynamicEngineStartMonitor( m_scheduler, m_now );
    }

    bool interrupted() const;

    PushPullEventQueue & pushPullEventQueue() { return m_pushPullEventQueue; }
    
protected:
    enum State { NONE, STARTING, RUNNING, SHUTDOWN, DONE };
    using EndCycleListeners = std::vector<EndCycleListener*>;

    //assign ranks, returns maxrank
    void    preRun( DateTime start, DateTime end );
    void    postRun();

    void    runSim( DateTime end );
    void    runRealtime( DateTime end );

    void    processPendingPushEvents( std::vector<PushGroup*> & dirtyGroups );
    void    processPushEventQueue( PushEvent * events, std::vector<PushGroup*> & dirtyGroups );

    void    processEndCycle();

    struct Settings
    {
        Settings( const Dictionary & );

        TimeDelta queueWaitTime;
        bool      realtime;
    };

    //dialect specific methods
    virtual void dialectUnlockGIL() noexcept {}
    virtual void dialectLockGIL() noexcept  {}

    //Main engine
    CycleStepTable    m_cycleStepTable;
    Scheduler         m_scheduler;
    DateTime          m_now;
    State             m_state;
    uint64_t          m_cycleCount;
    EndCycleListeners m_endCycleListeners;
    DateTime          m_startTime;
    DateTime          m_endTime;
    PendingPushEvents m_pendingPushEvents;
    Settings          m_settings;
    bool              m_inRealtime;
    int               m_initSignalCount;

    PushEventQueue     m_pushEventQueue;
    //This queue is managed entirely from the PushPullInputAdapter
    PushPullEventQueue m_pushPullEventQueue;

    std::exception_ptr                m_exception_ptr;
    std::mutex                        m_exception_mutex;
    std::unique_ptr<csp::Profiler>    m_profiler;

};

inline Scheduler::Handle RootEngine::reserveSchedulerHandle()
{
    return m_scheduler.reserveHandle();
}

inline Scheduler::Handle RootEngine::scheduleCallback( TimeDelta delta, Scheduler::Callback cb )
{
    return scheduleCallback( reserveSchedulerHandle(), m_now + delta, std::move( cb ) );
}

inline Scheduler::Handle RootEngine::scheduleCallback( DateTime time, Scheduler::Callback cb )
{
    return scheduleCallback( reserveSchedulerHandle(), time, std::move( cb ) );
}

inline Scheduler::Handle RootEngine::scheduleCallback( Scheduler::Handle reservedHandle, TimeDelta delta, Scheduler::Callback cb )
{
    return scheduleCallback( m_now + delta, std::move( cb ) );
}

inline Scheduler::Handle RootEngine::scheduleCallback( Scheduler::Handle reservedHandle, DateTime time, Scheduler::Callback cb )
{
    if( unlikely( time < m_now ) )
        CSP_THROW( ValueError, "Cannot schedule event in the past.  new time: " << time << " now: " << m_now );

    return m_scheduler.scheduleCallback( reservedHandle, time, std::move( cb ) ); 
}

inline Scheduler::Handle RootEngine::rescheduleCallback( Scheduler::Handle id, csp::DateTime time )
{
    if( unlikely( time < m_now ) )
        CSP_THROW( ValueError, "Cannot schedule event in the past. new time: " << time << " now: " << m_now );

    return m_scheduler.rescheduleCallback( id, time );
}

inline void RootEngine::cancelCallback( Scheduler::Handle handle )
{
    m_scheduler.cancelCallback( handle );
}

inline bool RootEngine::scheduleEndCycleListener( EndCycleListener * l )
{
    if( !l -> isDirty() )
    {
        l -> setDirtyFlag();
        m_endCycleListeners.emplace_back( l );
        return true;
    }
    return false;
}

};

#endif
