# C API Adapter Example

This example demonstrates how to implement CSP adapters using the **stable C ABI interface**. The C API provides a language-agnostic way to create input adapters, output adapters, and adapter managers that can be compiled separately from CSP.

## Why Use the C API?

- **ABI Stability**: The C API is designed to be stable across CSP versions
- **Language Flexibility**: C code can be called from Rust, Go, Zig, or any language with C FFI
- **Separate Compilation**: Adapters can be built as standalone libraries
- **Simpler Build**: No need to link against C++ STL or match compiler versions

## Architecture

The C API uses a capsule-based pattern where:

1. **C code** creates VTable structs with callbacks and wraps them in Python capsules
1. **CSP bridge functions** consume these capsules and create native adapter objects
1. **Python wiring** connects everything to the CSP graph

```
+-------------------+     +----------------------+     +------------------+
|   Your C Code     | --> |   Python Capsule     | --> |  CSP Bridge      |
| (VTable + state)  |     | (wraps VTable ptr)   |     | (creates adapter)|
+-------------------+     +----------------------+     +------------------+
```

## Key Concepts

### Output Adapter

Receives data from the CSP graph and sends it to external systems (files, network, databases).

```c
typedef struct CCspOutputAdapterVTable {
    void* user_data;
    void (*start)(void* user_data, CCspEngineHandle engine,
                  CCspDateTime start_time, CCspDateTime end_time);
    void (*stop)(void* user_data);
    void (*execute)(void* user_data, CCspEngineHandle engine, CCspInputHandle input);
    void (*destroy)(void* user_data);
} CCspOutputAdapterVTable;
```

### Push Input Adapter

Pushes data into the CSP graph from external sources (threads, callbacks).

```c
typedef struct CCspPushInputAdapterVTable {
    void* user_data;
    void (*start)(void* user_data, CCspEngineHandle engine,
                  CCspPushInputAdapterHandle adapter, ...);
    void (*stop)(void* user_data);
    void (*destroy)(void* user_data);
} CCspPushInputAdapterVTable;
```

### Adapter Manager

Coordinates the lifecycle of multiple adapters (shared connections, time slicing).

```c
typedef struct CCspAdapterManagerVTable {
    void* user_data;
    const char* name;
    void (*start)(void* user_data, CCspDateTime start_time, CCspDateTime end_time);
    void (*stop)(void* user_data);
    void (*destroy)(void* user_data);
} CCspAdapterManagerVTable;
```

## Building

```bash
# From this directory
hatch-build --hooks-only -t wheel

# Or install in development mode
pip install -e .
```

## Python Wiring Pattern

The key to integrating C API adapters with CSP is the **bridge function pattern**. CSP provides bridge functions that consume capsules and create native adapters:

- `_cspimpl._c_api_push_input_adapter` - For push input adapters
- `_cspimpl._c_api_output_adapter` - For output adapters

### Input Adapter Wiring

```python
from csp.impl.__cspimpl import _cspimpl
from csp.impl.wiring import input_adapter_def

from . import _my_native_module

def _create_my_input_adapter(mgr, engine, pytype, push_mode, scalars):
    """Bridge function called by input_adapter_def."""
    # Extract parameters from scalars
    interval_ms = scalars[1] if len(scalars) > 1 else 100

    # Create the VTable capsule using your C function
    capsule = _my_native_module._my_input_adapter(interval_ms=interval_ms)

    # Pass to CSP bridge which creates the actual adapter
    # Args: (capsule, push_group_or_none)
    return _cspimpl._c_api_push_input_adapter(
        mgr, engine, pytype, push_mode, (capsule, None)
    )

# Define the adapter
my_input_adapter = input_adapter_def(
    "my_input_adapter",
    _create_my_input_adapter,
    ts["T"],
    typ="T",
    interval_ms=int,
)
```

### Output Adapter Wiring

```python
from csp.impl.__cspimpl import _cspimpl
from csp.impl.wiring import output_adapter_def

def _create_my_output_adapter(mgr, engine, scalars):
    """Bridge function called by output_adapter_def."""
    # Extract parameters from scalars
    prefix = scalars[0] if scalars else ""

    # Create the VTable capsule
    capsule = _my_native_module._my_output_adapter(prefix=prefix)

    # Pass to CSP bridge
    # Args: (input_type, capsule)
    return _cspimpl._c_api_output_adapter(mgr, engine, (int, capsule))

# Define the adapter
my_output_adapter = output_adapter_def(
    "my_output_adapter",
    _create_my_output_adapter,
    input=ts["T"],
    prefix=str,
)
```

## Usage from Python

```python
from datetime import timedelta

import csp
from csp.utils.datetime import utc_now

from exampleadapter import example_input, example_output

@csp.graph
def my_graph():
    # Create input that generates integers every 100ms
    data = example_input(int, interval_ms=100)

    # Output to stdout with a prefix
    example_output(data, prefix="[MyApp] ")

csp.run(my_graph, starttime=utc_now(), endtime=timedelta(seconds=5))
```

## API Headers

The C API is defined in these headers (installed with CSP):

- `<csp/engine/c/CspType.h>` - Type definitions
- `<csp/engine/c/CspValue.h>` - Value types
- `<csp/engine/c/CspTime.h>` - DateTime/TimeDelta
- `<csp/engine/c/CspError.h>` - Error handling
- `<csp/engine/c/CspStruct.h>` - Struct access
- `<csp/engine/c/CspDictionary.h>` - Dictionary access
- `<csp/engine/c/OutputAdapter.h>` - Output adapter interface
- `<csp/engine/c/InputAdapter.h>` - Input adapter interface
- `<csp/engine/c/AdapterManager.h>` - Adapter manager interface

## Python Integration Headers

Helper functions for creating Python capsules:

- `<csp/python/c/PyOutputAdapter.h>` - Output adapter capsule helpers
- `<csp/python/c/PyInputAdapter.h>` - Input adapter capsule helpers
- `<csp/python/c/PyAdapterManager.h>` - Adapter manager capsule helpers

## See Also

- [C-APIs Reference](../../../docs/wiki/api-references/C-APIs.md)
- [Write C API Adapters Guide](../../../docs/wiki/how-tos/Write-C-API-Adapters.md)
- [Rust Adapter Example](../5_c_api_adapter_rust/) - Using the C API from Rust
