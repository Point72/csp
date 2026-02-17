#ifndef _IN_CSP_EXAMPLE_COUNTER_OUTPUT_ADAPTER_H
#define _IN_CSP_EXAMPLE_COUNTER_OUTPUT_ADAPTER_H

#include <csp/engine/OutputAdapter.h>

namespace csp::adapters::counter
{

/**
 * CounterOutputAdapter - A simple output adapter that logs values to stdout.
 *
 * This demonstrates the basic pattern for an OutputAdapter:
 * - Inherits from OutputAdapter
 * - Implements executeImpl() which is called whenever the input ticks
 * - Uses input()->lastValueTyped<T>() to get the current value
 */
class CounterOutputAdapter final : public OutputAdapter
{
public:
    CounterOutputAdapter( Engine * engine, CspTypePtr & type );
    ~CounterOutputAdapter() = default;

    void executeImpl() override;
    const char * name() const override { return "CounterOutputAdapter"; }

private:
    CspTypePtr m_type;
};

}

#endif
