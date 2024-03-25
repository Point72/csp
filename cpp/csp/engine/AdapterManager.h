#ifndef _IN_CSP_ADAPTER_MANAGER_H
#define _IN_CSP_ADAPTER_MANAGER_H

#include <csp/core/Time.h>
#include <csp/engine/InputAdapter.h>
#include <csp/engine/PushInputAdapter.h>
#include <csp/engine/RootEngine.h>
#include <csp/engine/StatusAdapter.h>
#include <optional>
#include <string>

namespace csp
{

class AdapterManager;

class Engine;

class ManagedSimInputAdapter : public InputAdapter
{
public:
    ManagedSimInputAdapter( csp::Engine *engine, const CspTypePtr &type, AdapterManager *manager, PushMode pushMode );

    template< typename T >
    bool pushTick( const T &value );

    template< typename T >
    bool pushNullTick();

private:
    template<typename T>
    void scheduleDelayedValue(std::optional<T>&& value);

    template< typename T >
    bool consumeDelayedValue();

    AdapterManager * m_manager;
    std::uint64_t m_lastCycleCount;
};

template< typename T >
inline bool ManagedSimInputAdapter::pushTick( const T &value )
{
    if( pushMode() == PushMode::NON_COLLAPSING )
    {
        auto cycleCount = rootEngine() -> cycleCount();
        if( m_lastCycleCount == cycleCount || !consumeTick( value ) )
        {
            //for non-collapsing data with duplicate timestamps,
            //we schedule the data to further engine cycles
            rootEngine() -> scheduleCallback( rootEngine() -> now(), [ this, value ]()  -> const InputAdapter*
                                              {
                                                  auto cycleCount = rootEngine() -> cycleCount();
                                                  if(m_lastCycleCount == cycleCount)
                                                      return this;

                                                  m_lastCycleCount = cycleCount;
                                                  consumeTick(value);
                                                  return nullptr;
                                              } );
        }
        m_lastCycleCount = cycleCount;
    }
    else
        consumeTick( value );

    //Note we always return true since we rescheduled our own timer above if need be
    return true;
}

template< typename T >
bool ManagedSimInputAdapter::pushNullTick()
{
    if( pushMode() == PushMode::NON_COLLAPSING )
    {
        auto cycleCount = rootEngine() -> cycleCount();

        if( m_lastCycleCount == cycleCount )
        {
            rootEngine() -> scheduleCallback( rootEngine() -> now(), [ this ]() -> const InputAdapter*
            {
                auto cycleCount = rootEngine() -> cycleCount();
                if( m_lastCycleCount == cycleCount )
                    return this;

                m_lastCycleCount = cycleCount;
                return nullptr;
            } );
        }
        m_lastCycleCount = cycleCount;
    }

    return true;
}

class AdapterManager : public EngineOwned
{
public:
    AdapterManager( csp::Engine * );
    virtual ~AdapterManager();

    virtual const char *name() const = 0;

    //derivations should call base start from their start call
    virtual void start( DateTime starttime, DateTime endtime );
    virtual void stop();

    //for sim inputs, this should process all entries with time "time" and return the time of the next
    //available tick, or return DateTime::NONE() if there is no more data.  Initial call will be with starttime
    //subsequent calls will be with the previously returned DateTime
    virtual DateTime processNextSimTimeSlice( DateTime time ) = 0;

    Engine * engine()               { return m_engine; }
    const Engine * engine() const   { return m_engine; }

    RootEngine * rootEngine()             { return m_engine -> rootEngine(); }
    const RootEngine * rootEngine() const { return m_engine -> rootEngine(); }

    DateTime starttime() const      { return m_starttime; }
    DateTime endtime() const        { return m_endtime; }
    
    void setStarted()               { m_started = true; }
    bool started() const            { return m_started; }

    StatusAdapter *createStatusAdapter( CspTypePtr &type, PushMode pushMode );
    void pushStatus( int64_t level, int64_t errCode, const std::string &errMsg, PushBatch *batch = nullptr ) const;

protected:
    //for adapters that want status adapter synced to a PushGroup
    virtual PushGroup * statusPushGroup() { return nullptr; }

    void scheduleTimerCB( DateTime next );
    
    void processSimTimerCB();

    csp::Engine   * m_engine;
    DateTime        m_starttime;
    DateTime        m_endtime;
    StatusAdapter * m_statusAdapter;
    bool            m_started;
};

inline void AdapterManager::scheduleTimerCB( DateTime next )
{
    try
    {
        rootEngine() -> scheduleCallback( next, [ this ](){ processSimTimerCB(); return nullptr; } );
    }
    catch( const ValueError &err )
    {
        CSP_THROW( ValueError, "AdapterManager " << name() << " scheduler error: " << err.description() );
    }

}

}

#endif
