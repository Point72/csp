#ifndef _IN_CSP_EXAMPLE_COUNTER_INPUT_ADAPTER_H
#define _IN_CSP_EXAMPLE_COUNTER_INPUT_ADAPTER_H

#include <csp/engine/PushInputAdapter.h>

namespace csp::adapters::counter
{

/**
 * CounterInputAdapter - A simple push input adapter that receives counter values
 * from the CounterAdapterManager's background thread.
 *
 * This demonstrates the basic pattern for a PushInputAdapter:
 * - Inherits from PushInputAdapter
 * - Receives data via pushTick() called from a background thread
 * - Data is automatically marshaled to the engine thread
 */
class CounterInputAdapter final : public PushInputAdapter
{
public:
    CounterInputAdapter( Engine * engine, CspTypePtr & type, PushMode pushMode, PushGroup * group );
    ~CounterInputAdapter() = default;
};

}

#endif
