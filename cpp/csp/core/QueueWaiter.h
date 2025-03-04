#ifndef _IN_CSP_CORE_QUEUEBLOCKINGWAIT_H
#define _IN_CSP_CORE_QUEUEBLOCKINGWAIT_H

#include <mutex>
#include <condition_variable>
#include <csp/core/Time.h>
#include <csp/core/System.h>

namespace csp
{

class TimeDelta;

class QueueWaiter
{
public:
    QueueWaiter() : m_eventsPending( false )
    {}

    void notify()
    {
        std::lock_guard<std::mutex> guard( m_lock );
        if( !m_eventsPending )
            m_condition.notify_one();
        m_eventsPending = true;

    }

    bool wait( TimeDelta maxWaitTime )
    {
        std::unique_lock<std::mutex> lock( m_lock );
        if( !m_eventsPending && maxWaitTime.asNanoseconds() > 0 )
            m_condition.wait_for( lock, std::chrono::nanoseconds( maxWaitTime.asNanoseconds() ), [this]() { return m_eventsPending; } );

        bool rv = m_eventsPending;
        m_eventsPending = false;
        return rv;
    }

private:
    std::mutex              m_lock;
    std::condition_variable m_condition;
    bool                    m_eventsPending;
};

}

#endif
