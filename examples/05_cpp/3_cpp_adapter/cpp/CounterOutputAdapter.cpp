#include "CounterOutputAdapter.h"
#include <csp/engine/TimeSeriesProvider.h>
#include <iostream>

namespace csp::adapters::counter
{

CounterOutputAdapter::CounterOutputAdapter( Engine * engine, CspTypePtr & type )
    : OutputAdapter( engine ), m_type( type )
{
}

void CounterOutputAdapter::executeImpl()
{
    // Get the last value from the input time series
    // For this example, we assume int64_t type
    int64_t value = input() -> lastValueTyped<int64_t>();
    std::cout << "[CounterOutputAdapter] Received value: " << value << std::endl;
}

}
