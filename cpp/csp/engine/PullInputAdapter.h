#ifndef _IN_CSP_ENGINE_PULLINPUTADAPTER_H
#define _IN_CSP_ENGINE_PULLINPUTADAPTER_H

#include <csp/core/Time.h>
#include <csp/engine/Engine.h>
#include <csp/engine/InputAdapter.h>

namespace csp
{

template<typename T>
class PullInputAdapter : public InputAdapter
{
public:
    PullInputAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode ) : InputAdapter( engine, type, pushMode ) 
    {
    }

    virtual void start( DateTime start, DateTime end ) override;
    virtual void stop() final;
    virtual bool next( DateTime & time, T & value ) = 0;

private:
    virtual void stopAdapter() {}

    bool processNext();

    Scheduler::Handle m_timerHandle;
    T                 m_nextValue;
};

template< typename T >
void PullInputAdapter<T>::start( DateTime start, DateTime end )
{
    DateTime t;
    if( next( t, m_nextValue ) )
        m_timerHandle = rootEngine() -> scheduleCallback( t, [this]() { return processNext() ? nullptr : this; } );
}

template< typename T >
void PullInputAdapter<T>::stop()
{
    rootEngine() -> cancelCallback( m_timerHandle );
    stopAdapter();
}

template< typename T >
bool PullInputAdapter<T>::processNext()
{
    bool consumed = consumeTick( m_nextValue );
    assert( consumed );

    bool rv;

    DateTime t;
    while( consumed && ( rv = next( t, m_nextValue ) ) && 
           t == rootEngine() -> now() )
    {
        consumed = consumeTick( m_nextValue );
    }

    //non-collapsing on same engine tick, rerun next engine cycle
    if( !consumed )
        return false;

    if( rv )
        m_timerHandle = rootEngine() -> scheduleCallback( t, [this]() { return processNext() ? nullptr : this; } );

    return true;
}

};

#endif
