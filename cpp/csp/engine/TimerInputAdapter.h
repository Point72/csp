#ifndef _IN_CSP_ENGINE_TIMERINPUTADAPTER_H
#define _IN_CSP_ENGINE_TIMERINPUTADAPTER_H

#include <csp/engine/PullInputAdapter.h>

namespace csp
{

template<typename T>
class TimerInputAdapter : public PullInputAdapter<T>
{
public:
    TimerInputAdapter( Engine * engine, CspTypePtr & type, TimeDelta interval, 
                       const T & value, bool allowDeviation ) : PullInputAdapter<T>( engine, type, PushMode::LAST_VALUE ),
                                                                m_interval( interval ),
                                                                m_value( value ),
                                                                m_allowDeviation( allowDeviation )
    {
    }

    void start( DateTime start, DateTime end ) override
    {
        //note first tick will be at start + interval
        m_time = start;
        PullInputAdapter<T>::start( start, end );
    }

    bool next( DateTime & t, T & value ) override
    {
        if( m_allowDeviation && this -> rootEngine() -> inRealtime() )
            m_time = DateTime::now() + m_interval;
        else
            m_time += m_interval;

        t     = m_time;
        value = m_value;
        return true;
    }

private:
    TimeDelta m_interval;
    DateTime  m_time;
    T         m_value;
    bool      m_allowDeviation;
};

};

#endif
