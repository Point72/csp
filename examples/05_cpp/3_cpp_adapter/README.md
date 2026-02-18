# Custom C++ Adapter

> [!WARNING]
> **Internal API - Use with Caution**
>
> This example demonstrates CSP's internal C++ adapter pattern. The C++ API is
> **not stable** and may change without notice. For production adapters, use
> the stable [C API](../4_c_api_adapter/README.md) instead.
>
> This example is useful for understanding how CSP adapters work internally,
> or if you need features not yet exposed through the C API.

## Overview

This example demonstrates how to create a complete C++ adapter ecosystem for CSP,
including:

- **`CounterAdapterManager`** - Manages adapter lifecycle and coordinates input/output
- **`CounterInputAdapter`** - A push input adapter that generates sequential counter values
- **`CounterOutputAdapter`** - An output adapter that logs values to stdout

The adapter generates counter values at a configurable interval, demonstrating
the push input adapter pattern commonly used for real-time data sources.

## Key Concepts

### AdapterManager

The `AdapterManager` is responsible for:

- Managing the lifecycle of all adapters it creates
- Coordinating start/stop across adapters
- Providing factory methods for creating input and output adapters
- Managing background threads (for real-time push adapters)

```cpp
class CounterAdapterManager final : public csp::AdapterManager
{
public:
    void start( DateTime starttime, DateTime endtime ) override;
    void stop() override;

    PushInputAdapter * getInputAdapter( ... );
    OutputAdapter * getOutputAdapter( ... );
};
```

### PushInputAdapter

For real-time data sources, use `PushInputAdapter` which allows pushing data
from background threads:

```cpp
class CounterInputAdapter final : public PushInputAdapter
{
    // Data is pushed via pushTick() from a background thread
};

// In the adapter manager's background thread:
void CounterAdapterManager::pushValue( int64_t value )
{
    PushBatch batch( rootEngine() );
    m_inputAdapter->pushTick( value, &batch );
}
```

### OutputAdapter

Output adapters receive data from the graph and perform side effects:

```cpp
class CounterOutputAdapter final : public OutputAdapter
{
    void executeImpl() override
    {
        int64_t value = input()->lastValueTyped<int64_t>();
        std::cout << "Received: " << value << std::endl;
    }
};
```

### Python Bindings

Use CSP's registration macros to expose C++ adapters to Python:

```cpp
REGISTER_ADAPTER_MANAGER( _counter_adapter_manager, create_counter_adapter_manager );
REGISTER_INPUT_ADAPTER(   _counter_input_adapter,   create_counter_input_adapter );
REGISTER_OUTPUT_ADAPTER(  _counter_output_adapter,  create_counter_output_adapter );
```

## Building

```bash
cd examples/05_cpp/3_cpp_adapter
python setup.py build build_ext --inplace
```

## Usage

After building, you can use the adapter in Python:

```python
import csp
from csp.utils.datetime import utc_now

from datetime import timedelta

from counteradapter import CounterAdapterManager

@csp.graph
def my_graph():
    # Create manager with 100ms interval, max 10 counts
    mgr = CounterAdapterManager(interval_ms=100, max_count=10)

    # Subscribe to counter values
    data = mgr.subscribe()

    # Print values
    csp.print("Counter", data)

    # Also publish to output adapter
    mgr.publish(data)

# Run for 2 seconds in realtime mode
csp.run(my_graph, starttime=utc_now(), endtime=timedelta(seconds=2), realtime=True)
```

Or run the example directly:

```bash
python -m counteradapter
```

## API Reference

### CounterAdapterManager

| Parameter     | Type  | Default | Description                                    |
| ------------- | ----- | ------- | ---------------------------------------------- |
| `interval_ms` | `int` | `1000`  | Interval between counter ticks in milliseconds |
| `max_count`   | `int` | `0`     | Maximum count before stopping (0 = unlimited)  |

### Methods

- **`subscribe() -> csp.ts[int]`** - Subscribe to counter values
- **`publish(data: csp.ts[int])`** - Publish values to the output adapter

## CSP Internal Headers Used

This example uses the following internal CSP headers:

```cpp
#include <csp/engine/AdapterManager.h>     // Base AdapterManager class
#include <csp/engine/PushInputAdapter.h>   // PushInputAdapter for real-time data
#include <csp/engine/OutputAdapter.h>      // Base OutputAdapter class
#include <csp/engine/Dictionary.h>         // Configuration dictionary
#include <csp/python/PyAdapterManagerWrapper.h>  // Python bindings
#include <csp/python/PyInputAdapterWrapper.h>    // Input adapter bindings
#include <csp/python/PyOutputAdapterWrapper.h>   // Output adapter bindings
#include <csp/python/InitHelper.h>         // Registration macros
```

## See Also

- [C API Adapter](../4_c_api_adapter/README.md) - Stable C API for adapters (recommended)
- [Rust C API Adapter](../5_c_api_adapter_rust/README.md) - Rust adapter using the C API
- [CSP Adapters Documentation](../../../docs/wiki/how-tos/Write-Adapters.md)
