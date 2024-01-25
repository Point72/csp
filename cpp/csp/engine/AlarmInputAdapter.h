#ifndef _IN_CSP_ENGINE_ALARMINPUTADAPTER_H
#define _IN_CSP_ENGINE_ALARMINPUTADAPTER_H

#include <csp/engine/InputAdapter.h>
#include <unordered_set>

namespace csp
{

template<typename T>
class AlarmInputAdapter final : public InputAdapter
{
public:
    AlarmInputAdapter( Engine * engine, CspTypePtr & type ) : InputAdapter( engine, type, PushMode::NON_COLLAPSING )
    {
    }

    void start( DateTime start, DateTime end ) override {}
    void stop() override
    {
        for( auto & handle : m_pendingHandles )
            rootEngine() -> cancelCallback( handle );
        m_pendingHandles.clear();
    }

    Scheduler::Handle scheduleAlarm( TimeDelta delta, const T & value )
    {
        return scheduleAlarm( rootEngine() -> now() + delta, value );
    }

    Scheduler::Handle scheduleAlarm( DateTime time, const T & value )
    {
        auto handle = rootEngine() -> reserveSchedulerHandle();
        auto it = m_pendingHandles.insert( m_pendingHandles.end(), handle );
        handle = rootEngine() -> scheduleCallback( handle, time, 
                                                   [this, value, it]() -> const InputAdapter *
                                                   {
                                                       if( !this -> consumeTick<T>( value ) )
                                                           return this;

                                                       m_pendingHandles.erase( it );
                                                       return nullptr;
                                                   } );
        (*it) = handle;
        return handle;
    }

    Scheduler::Handle rescheduleAlarm( Scheduler::Handle handle, TimeDelta delta )
    {
        return rescheduleAlarm( handle, rootEngine() -> now() + delta );
    }

    Scheduler::Handle rescheduleAlarm( Scheduler::Handle handle, DateTime time )
    {
        return rootEngine() -> rescheduleCallback( handle, time );
    }

    void cancelAlarm( Scheduler::Handle handle )
    {
        rootEngine() -> cancelCallback( handle );
    }

private:
    std::list<Scheduler::Handle> m_pendingHandles;
};

};

#endif
