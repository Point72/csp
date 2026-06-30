#ifndef _IN_CSP_EXAMPLE_COUNTER_ADAPTER_MANAGER_H
#define _IN_CSP_EXAMPLE_COUNTER_ADAPTER_MANAGER_H

#include <csp/engine/AdapterManager.h>
#include <csp/engine/Dictionary.h>
#include <csp/engine/PushInputAdapter.h>
#include <thread>
#include <atomic>

namespace csp::adapters::counter
{

class CounterInputAdapter;
class CounterOutputAdapter;

/**
 * CounterAdapterManager - A simple example adapter manager that demonstrates
 * the basic pattern for creating custom adapters in csp.
 *
 * This adapter manager:
 * - Controls the lifecycle of CounterInputAdapter and CounterOutputAdapter
 * - Manages a background thread that periodically generates counter values
 * - Demonstrates PushInputAdapter and OutputAdapter patterns
 */
class CounterAdapterManager final : public csp::AdapterManager
{
public:
    CounterAdapterManager( csp::Engine * engine, const Dictionary & properties );
    ~CounterAdapterManager();

    const char * name() const override { return "CounterAdapterManager"; }

    void start( DateTime starttime, DateTime endtime ) override;
    void stop() override;

    // For sim inputs - we return NONE since this is a realtime adapter
    DateTime processNextSimTimeSlice( DateTime time ) override { return DateTime::NONE(); }

    // Factory methods for creating adapters
    PushInputAdapter * getInputAdapter( CspTypePtr & type, PushMode pushMode, const Dictionary & properties );
    OutputAdapter * getOutputAdapter( CspTypePtr & type, const Dictionary & properties );

    // Internal method called by the push thread
    void pushValue( int64_t value );

private:
    void runPushThread();

    // Configuration from properties
    int64_t m_intervalMs;      // Interval between pushes in milliseconds
    int64_t m_maxCount;        // Maximum count before stopping (0 = unlimited)

    // Thread management
    std::unique_ptr<std::thread> m_pushThread;
    std::atomic<bool> m_running;

    // Push group for coordinating input adapters
    PushGroup m_pushGroup;

    // Registered adapters
    CounterInputAdapter * m_inputAdapter;
    CounterOutputAdapter * m_outputAdapter;
};

}

#endif
