## Table of Contents

- [Table of Contents](#table-of-contents)
- [Writing C API Adapters](#writing-c-api-adapters)
  - [Overview](#overview)
  - [When to Use the C API](#when-to-use-the-c-api)
  - [The VTable Pattern](#the-vtable-pattern)
  - [Writing an Output Adapter in C](#writing-an-output-adapter-in-c)
    - [Step 1: Define Your State](#step-1-define-your-state)
    - [Step 2: Implement Callbacks](#step-2-implement-callbacks)
    - [Step 3: Create the Factory Function](#step-3-create-the-factory-function)
  - [Writing a Push Input Adapter in C](#writing-a-push-input-adapter-in-c)
    - [Step 1: Define State with Threading](#step-1-define-state-with-threading)
    - [Step 2: Implement the Data Source Thread](#step-2-implement-the-data-source-thread)
    - [Step 3: Implement Lifecycle Callbacks](#step-3-implement-lifecycle-callbacks)
    - [Step 4: Create Factory Function](#step-4-create-factory-function)
  - [Writing an Adapter Manager in C](#writing-an-adapter-manager-in-c)
    - [Why Use an Adapter Manager](#why-use-an-adapter-manager)
    - [Adapter Manager VTable](#adapter-manager-vtable)
    - [Example: Managed Adapter](#example-managed-adapter)
  - [Building and Linking](#building-and-linking)
  - [Python Integration](#python-integration)
    - [Create Python Bindings (C code)](#create-python-bindings-c-code)
    - [Create Python Wrapper with Bridge Functions](#create-python-wrapper-with-bridge-functions)
    - [CSP Bridge Functions](#csp-bridge-functions)
    - [Managed Adapter Python Wrapper](#managed-adapter-python-wrapper)
    - [Use Managed Adapters in Your Graph](#use-managed-adapters-in-your-graph)
    - [Use in Your Graph](#use-in-your-graph)
- [See Also](#see-also)

## Writing C API Adapters

CSP provides a C API for writing adapters that can be compiled separately from CSP and loaded at runtime. This enables:

- Writing adapters in any language with C FFI (C, Rust, Go, etc.)
- Distributing adapters as separate packages
- ABI stability across CSP versions

### Overview

The C API uses a **VTable (Virtual Table) pattern** to define adapters. A VTable is a struct containing function pointers that CSP calls at the appropriate times during the adapter lifecycle:

```raw
+-------------------------------------------------------------+
|                      CSP Engine (C++)                       |
|                                                             |
|  1. Calls vtable.start() when graph starts                  |
|  2. Calls vtable.execute() when input ticks (output adapter)|
|  3. Calls vtable.stop() when graph stops                    |
|  4. Calls vtable.destroy() to clean up                      |
|                                                             |
+---------------------+---------------------------------------+
                      | calls your functions
                      v
+-------------------------------------------------------------+
|                    Your C Adapter                           |
|                                                             |
|  - Allocate state struct                                    |
|  - Implement callback functions                             |
|  - Return VTable with function pointers                     |
|                                                             |
+-------------------------------------------------------------+
```

### When to Use the C API

Use the C API when you need:

| Use Case                            | Example                                         |
| ----------------------------------- | ----------------------------------------------- |
| **Separate compilation**            | Distribute Kafka adapter independently from CSP |
| **Non-Python languages**            | Write an adapter in Rust or Go                  |
| **Performance-critical code**       | Avoid Python overhead in hot paths              |
| **Third-party library integration** | Wrap a C library directly                       |

For simpler use cases, consider using Python adapters instead (see [Write Output Adapters](Write-Output-Adapters.md) and [Write Realtime Input Adapters](Write-Realtime-Input-Adapters.md)).

### The VTable Pattern

Every C adapter consists of:

1. **A state struct** - holds your adapter's data
1. **Callback functions** - implement the adapter logic
1. **A factory function** - allocates state and returns a VTable

Here's the VTable structure for output adapters:

```c
typedef struct CCspOutputAdapterVTable {
    void* user_data;  // Pointer to your state struct

    // Called when graph starts
    void (*start)(void* user_data, CCspEngineHandle engine,
                  CCspDateTime start_time, CCspDateTime end_time);

    // Called when graph stops
    void (*stop)(void* user_data);

    // Called when input ticks (REQUIRED)
    void (*execute)(void* user_data, CCspEngineHandle engine,
                    CCspInputHandle input);

    // Called to clean up (REQUIRED)
    void (*destroy)(void* user_data);
} CCspOutputAdapterVTable;
```

### Writing an Output Adapter in C

Let's write a simple output adapter that logs values to stdout.

#### Step 1: Define Your State

```c
#include <csp/engine/c/OutputAdapter.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char* prefix;       // Prefix for log messages
    FILE* output;       // Where to write
} LogAdapterState;
```

#### Step 2: Implement Callbacks

```c
static void log_adapter_start(void* user_data, CCspEngineHandle engine,
                               CCspDateTime start_time, CCspDateTime end_time)
{
    LogAdapterState* state = (LogAdapterState*)user_data;
    fprintf(state->output, "[LogAdapter] Started\n");
}

static void log_adapter_stop(void* user_data)
{
    LogAdapterState* state = (LogAdapterState*)user_data;
    fprintf(state->output, "[LogAdapter] Stopped\n");
}

static void log_adapter_execute(void* user_data, CCspEngineHandle engine,
                                 CCspInputHandle input)
{
    LogAdapterState* state = (LogAdapterState*)user_data;

    // Get the current time
    CCspDateTime now = ccsp_engine_now(engine);

    // Get the input value based on type
    CCspType type = ccsp_input_get_type(input);

    switch (type) {
        case CCSP_TYPE_STRING: {
            const char* data;
            size_t len;
            if (ccsp_input_get_last_string(input, &data, &len) == CCSP_OK) {
                fprintf(state->output, "%s[%lld] %.*s\n",
                        state->prefix, (long long)now, (int)len, data);
            }
            break;
        }
        case CCSP_TYPE_INT64: {
            int64_t val;
            if (ccsp_input_get_last_int64(input, &val) == CCSP_OK) {
                fprintf(state->output, "%s[%lld] %lld\n",
                        state->prefix, (long long)now, (long long)val);
            }
            break;
        }
        case CCSP_TYPE_DOUBLE: {
            double val;
            if (ccsp_input_get_last_double(input, &val) == CCSP_OK) {
                fprintf(state->output, "%s[%lld] %f\n",
                        state->prefix, (long long)now, val);
            }
            break;
        }
        default:
            fprintf(state->output, "%s[%lld] <unsupported type>\n",
                    state->prefix, (long long)now);
    }
}

static void log_adapter_destroy(void* user_data)
{
    LogAdapterState* state = (LogAdapterState*)user_data;
    if (state) {
        free(state->prefix);
        free(state);
    }
}
```

#### Step 3: Create the Factory Function

```c
CCspOutputAdapterVTable create_log_adapter(const char* prefix)
{
    CCspOutputAdapterVTable vtable = {0};

    // Allocate state
    LogAdapterState* state = malloc(sizeof(LogAdapterState));
    if (!state) {
        return vtable;  // Return zeroed vtable on error
    }

    // Initialize state
    state->output = stdout;
    state->prefix = prefix ? strdup(prefix) : strdup("");

    // Fill in the vtable
    vtable.user_data = state;
    vtable.start = log_adapter_start;
    vtable.stop = log_adapter_stop;
    vtable.execute = log_adapter_execute;
    vtable.destroy = log_adapter_destroy;

    return vtable;
}
```

### Writing a Push Input Adapter in C

Push input adapters are more complex because they push data from external threads into the CSP engine.

#### Step 1: Define State with Threading

```c
#include <csp/engine/c/InputAdapter.h>
#include <pthread.h>

typedef struct {
    int interval_ms;
    int running;
    int64_t counter;
    CCspPushInputAdapterHandle adapter;  // Handle for pushing data
    pthread_t thread;
} CounterAdapterState;
```

#### Step 2: Implement the Data Source Thread

```c
static void* counter_thread(void* arg)
{
    CounterAdapterState* state = (CounterAdapterState*)arg;

    while (state->running) {
        // Push the current value (thread-safe)
        ccsp_push_input_adapter_push_int64(state->adapter, state->counter, NULL);
        state->counter++;

        usleep(state->interval_ms * 1000);
    }

    return NULL;
}
```

#### Step 3: Implement Lifecycle Callbacks

```c
static void counter_start(void* user_data, CCspEngineHandle engine,
                          CCspPushInputAdapterHandle adapter,
                          CCspDateTime start_time, CCspDateTime end_time)
{
    CounterAdapterState* state = (CounterAdapterState*)user_data;

    // Save the adapter handle for pushing data
    state->adapter = adapter;
    state->running = 1;
    state->counter = 0;

    // Start the data thread
    pthread_create(&state->thread, NULL, counter_thread, state);
}

static void counter_stop(void* user_data)
{
    CounterAdapterState* state = (CounterAdapterState*)user_data;
    state->running = 0;
    pthread_join(state->thread, NULL);
}

static void counter_destroy(void* user_data)
{
    free(user_data);
}
```

#### Step 4: Create Factory Function

```c
CCspPushInputAdapterVTable create_counter_adapter(int interval_ms)
{
    CCspPushInputAdapterVTable vtable = {0};

    CounterAdapterState* state = malloc(sizeof(CounterAdapterState));
    if (!state) return vtable;

    memset(state, 0, sizeof(CounterAdapterState));
    state->interval_ms = interval_ms > 0 ? interval_ms : 100;

    vtable.user_data = state;
    vtable.start = counter_start;
    vtable.stop = counter_stop;
    vtable.destroy = counter_destroy;

    return vtable;
}
```

### Writing an Adapter Manager in C

For complex adapters like Kafka or WebSocket, you'll want to use an **Adapter Manager** to coordinate multiple adapters that share resources (connections, threads, configuration).

#### Why Use an Adapter Manager

| Scenario                 | Without Manager                 | With Manager              |
| ------------------------ | ------------------------------- | ------------------------- |
| Multiple output adapters | Each manages its own connection | Shared connection pool    |
| Start/stop lifecycle     | Each adapter independently      | Coordinated start/stop    |
| Status reporting         | No unified status               | Single status stream      |
| Configuration            | Duplicated across adapters      | Centralized configuration |

Adapter managers are used by CSP's built-in Kafka, Parquet, and WebSocket adapters.

#### Adapter Manager VTable

```c
typedef struct CCspAdapterManagerVTable {
    void* user_data;

    // REQUIRED: Return the name of this manager
    const char* (*name)(void* user_data);

    // REQUIRED: Process simulation time slice (return 0 for realtime-only)
    CCspDateTime (*process_next_sim_time_slice)(void* user_data, CCspDateTime time);

    // REQUIRED: Clean up resources
    void (*destroy)(void* user_data);

    // OPTIONAL: Called when graph starts
    void (*start)(void* user_data, CCspAdapterManagerHandle manager,
                  CCspDateTime start_time, CCspDateTime end_time);

    // OPTIONAL: Called when graph stops
    void (*stop)(void* user_data);

} CCspAdapterManagerVTable;
```

#### Example: Managed Adapter

This example shows a manager that coordinates multiple output adapters (like Kafka topics):

```c
#include <csp/engine/c/AdapterManager.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Shared state for all adapters in this manager */
typedef struct {
    char name[64];
    int is_started;
    int total_messages;
    CCspAdapterManagerHandle manager_handle;
} ManagedState;

/* State for each output adapter */
typedef struct {
    ManagedState* shared;  /* Points to manager's state */
    char topic[64];
    int messages_sent;
} OutputState;

/* Manager callbacks */
static const char* manager_name(void* user_data) {
    ManagedState* state = (ManagedState*)user_data;
    return state->name;
}

static void manager_start(void* user_data, CCspAdapterManagerHandle manager,
                          CCspDateTime start_time, CCspDateTime end_time) {
    ManagedState* state = (ManagedState*)user_data;
    state->manager_handle = manager;
    state->is_started = 1;
    printf("[%s] Manager started\n", state->name);

    /* Report status to the graph */
    ccsp_adapter_manager_push_status(manager, CCSP_STATUS_LEVEL_INFO, 0,
                                     "Manager started successfully");
}

static void manager_stop(void* user_data) {
    ManagedState* state = (ManagedState*)user_data;
    printf("[%s] Manager stopped. Total messages: %d\n",
           state->name, state->total_messages);
    state->is_started = 0;
}

static CCspDateTime manager_process_sim(void* user_data, CCspDateTime time) {
    /* Realtime-only adapter - return 0 */
    (void)user_data;
    (void)time;
    return 0;
}

static void manager_destroy(void* user_data) {
    free(user_data);
}

/* Create the adapter manager */
CCspAdapterManagerVTable create_my_manager(const char* name) {
    CCspAdapterManagerVTable vtable = {0};

    ManagedState* state = malloc(sizeof(ManagedState));
    if (!state) return vtable;

    memset(state, 0, sizeof(ManagedState));
    strncpy(state->name, name ? name : "MyManager", sizeof(state->name) - 1);

    vtable.user_data = state;
    vtable.name = manager_name;
    vtable.start = manager_start;
    vtable.stop = manager_stop;
    vtable.process_next_sim_time_slice = manager_process_sim;
    vtable.destroy = manager_destroy;

    return vtable;
}

/* Output adapter callbacks - uses shared state */
static void output_execute(void* user_data, CCspEngineHandle engine,
                           CCspInputHandle input) {
    OutputState* state = (OutputState*)user_data;
    if (!ccsp_input_is_valid(input)) return;

    /* Get value and log it */
    CCspType type = ccsp_input_get_type(input);
    printf("  [%s/%s] received type %d\n",
           state->shared->name, state->topic, type);

    state->messages_sent++;
    state->shared->total_messages++;
}

static void output_destroy(void* user_data) {
    free(user_data);
}

/* Create an output adapter that uses the manager's shared state */
CCspOutputAdapterVTable create_managed_output(ManagedState* shared, const char* topic) {
    CCspOutputAdapterVTable vtable = {0};

    OutputState* state = malloc(sizeof(OutputState));
    if (!state) return vtable;

    memset(state, 0, sizeof(OutputState));
    state->shared = shared;
    strncpy(state->topic, topic ? topic : "default", sizeof(state->topic) - 1);

    vtable.user_data = state;
    vtable.execute = output_execute;
    vtable.destroy = output_destroy;

    return vtable;
}
```

See [ExampleManagedAdapter.c](../../cpp/csp/adapters/c/example/ExampleManagedAdapter.c) for the complete implementation.

### Building and Linking

Your C adapter should be compiled as a shared library that links against the CSP C API:

```cmake
# CMakeLists.txt for your adapter
find_package(csp REQUIRED)

add_library(my_adapter SHARED
    my_adapter.c
)

target_link_libraries(my_adapter
    csp::csp_c_api
)

target_include_directories(my_adapter PRIVATE
    ${CSP_INCLUDE_DIRS}
)
```

Build with:

```bash
mkdir build && cd build
cmake ..
make
```

### Python Integration

To use your C adapter from Python, you need to:

1. Create a Python extension module that wraps your C functions and returns capsules
1. Write Python bridge functions that pass capsules to CSP's adapter bridge
1. Define adapters using `input_adapter_def` / `output_adapter_def`

#### Create Python Bindings (C code)

Your C extension should create capsules wrapping the VTables:

```c
#include <Python.h>
#include <csp/python/c/PyOutputAdapter.h>
#include <csp/python/c/PyInputAdapter.h>
#include "my_adapter.h"

// Output adapter - returns a capsule
static PyObject* create_log_adapter_py(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"prefix", NULL};
    const char* prefix = "";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|s", kwlist, &prefix)) {
        return NULL;
    }

    CCspOutputAdapterVTable vtable = create_log_adapter(prefix);
    if (!vtable.execute) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create adapter");
        return NULL;
    }

    // Create a capsule that CSP's bridge can consume
    return ccsp_py_create_output_adapter_capsule_owned(&vtable);
}

// Input adapter - returns a capsule
static PyObject* create_input_adapter_py(PyObject* self, PyObject* args, PyObject* kwargs)
{
    static char* kwlist[] = {"interval_ms", NULL};
    int interval_ms = 100;

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|i", kwlist, &interval_ms)) {
        return NULL;
    }

    CCspPushInputAdapterVTable vtable = create_my_input_adapter(interval_ms);
    if (!vtable.destroy) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create adapter");
        return NULL;
    }

    return ccsp_py_create_input_adapter_capsule_owned(&vtable);
}

static PyMethodDef methods[] = {
    {"_log_adapter", (PyCFunction)create_log_adapter_py, METH_VARARGS | METH_KEYWORDS, "Create log adapter"},
    {"_my_input_adapter", (PyCFunction)create_input_adapter_py, METH_VARARGS | METH_KEYWORDS, "Create input adapter"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "my_adapter", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_my_adapter(void) {
    return PyModule_Create(&module);
}
```

#### Create Python Wrapper with Bridge Functions

CSP provides bridge functions that consume capsules and create native adapters. These are essential for integrating C API adapters with CSP's wiring layer.

```python
# my_adapter.py
from csp import ts
from csp.impl.__cspimpl import _cspimpl
from csp.impl.wiring import input_adapter_def, output_adapter_def
from . import _my_adapter_impl  # Your C extension

def _create_input_bridge(mgr, engine, pytype, push_mode, scalars):
    """
    Bridge function for input adapters.

    The wiring layer calls this with:
    - mgr: adapter manager (or None)
    - engine: CSP engine
    - pytype: Python type for the timeseries
    - push_mode: PushMode enum value
    - scalars: tuple of scalar arguments from the adapter_def
    """
    # Extract your parameters from scalars
    # For standalone adapters: scalars[0] is typ, scalars[1] is interval_ms
    interval_ms = scalars[1] if len(scalars) > 1 else 100

    # Create the VTable capsule using your C function
    capsule = _my_adapter_impl._my_input_adapter(interval_ms=interval_ms)

    # Pass to CSP bridge which creates PushInputAdapterExtern
    # Arguments: (capsule, push_group_or_none)
    return _cspimpl._c_api_push_input_adapter(
        mgr, engine, pytype, push_mode, (capsule, None)
    )

def _create_output_bridge(mgr, engine, scalars):
    """
    Bridge function for output adapters.

    The wiring layer calls this with:
    - mgr: adapter manager (or None)
    - engine: CSP engine
    - scalars: tuple of scalar arguments from the adapter_def
    """
    # Extract your parameters
    prefix = scalars[0] if scalars else ""

    # Create the VTable capsule
    capsule = _my_adapter_impl._log_adapter(prefix=prefix)

    # Pass to CSP bridge which creates OutputAdapterExtern
    # Arguments: (input_type, capsule)
    return _cspimpl._c_api_output_adapter(mgr, engine, (int, capsule))

# Define input adapter
my_input = input_adapter_def(
    "my_input_adapter",
    _create_input_bridge,
    ts["T"],
    typ="T",
    interval_ms=int,
)

# Define output adapter
LogAdapter = output_adapter_def(
    "LogAdapter",
    _create_output_bridge,
    input=ts["T"],
    prefix=str,
)
```

#### CSP Bridge Functions

CSP provides three bridge functions in `_cspimpl`:

| Function                        | Purpose                                         | Arguments                                                 |
| ------------------------------- | ----------------------------------------------- | --------------------------------------------------------- |
| `_c_api_push_input_adapter`     | Creates PushInputAdapterExtern from capsule     | `(mgr, engine, pytype, push_mode, (capsule, push_group))` |
| `_c_api_output_adapter`         | Creates OutputAdapterExtern from capsule        | `(mgr, engine, (input_type, capsule))`                    |
| `_c_api_adapter_manager_bridge` | Converts C API manager to CSP-compatible format | `(engine, capsule)`                                       |

These functions:

1. Extract the VTable from the capsule
1. Create the appropriate native adapter (`PushInputAdapterExtern`, `OutputAdapterExtern`, or `AdapterManagerExtern`)
1. Transfer ownership of the VTable to prevent double-free
1. Return a wrapper object compatible with CSP's wiring layer

#### Managed Adapter Python Wrapper

For adapter managers, you need to create a Python class that wraps your C manager and its adapters:

```python
# managed_adapter.py
import csp
from csp import ts
from csp.impl.__cspimpl import _cspimpl
from csp.impl.pushadapter import PushGroup
from csp.impl.wiring import input_adapter_def, output_adapter_def
from . import _my_native_module  # Your C/Rust extension


class MyAdapterManager:
    """
    Python wrapper for a C API adapter manager.

    The adapter manager pattern allows coordinating multiple adapters that share:
    - Startup/shutdown coordination
    - Push groups for batched event processing
    - Common configuration
    """

    def __init__(self, prefix: str = ""):
        self._prefix = prefix
        self._push_group = PushGroup()
        self._properties = {"prefix": prefix}

    def subscribe(self, ts_type=int, interval_ms=100, push_mode=csp.PushMode.NON_COLLAPSING):
        """Create an input adapter managed by this manager."""
        return _managed_input_def(
            self, ts_type, interval_ms=interval_ms,
            push_mode=push_mode, push_group=self._push_group
        )

    def publish(self, x, prefix=None):
        """Create an output adapter managed by this manager."""
        return _managed_output_def(self, x, prefix=prefix or self._prefix)

    def _create(self, engine, memo):
        """
        Called by CSP wiring layer to create the native manager.

        This is the key integration point - it bridges the C API capsule
        to CSP's expected format.
        """
        # Create C API adapter manager capsule
        c_api_capsule = _my_native_module._my_adapter_manager(engine, self._properties)

        # Bridge to CSP format (returns AdapterManagerExtern* wrapped in capsule)
        return _cspimpl._c_api_adapter_manager_bridge(engine, c_api_capsule)


def _create_managed_input(mgr_capsule, engine, pytype, push_mode, scalars):
    """Bridge function for managed input adapters."""
    # Extract interval_ms from scalars
    interval_ms = 100
    for s in scalars:
        if isinstance(s, int) and not isinstance(s, bool):
            interval_ms = s
            break

    # Create VTable capsule
    capsule = _my_native_module._my_input_adapter(interval_ms=interval_ms)

    # Extract push group (last element if present)
    push_group = None
    if scalars and "PushGroup" in type(scalars[-1]).__name__:
        push_group = scalars[-1]

    # Pass manager capsule (not None) to bridge
    return _cspimpl._c_api_push_input_adapter(
        mgr_capsule, engine, pytype, push_mode, (capsule, push_group)
    )


def _create_managed_output(mgr_capsule, engine, scalars):
    """Bridge function for managed output adapters."""
    prefix = scalars[1] if len(scalars) > 1 else ""
    capsule = _my_native_module._my_output_adapter(prefix=prefix)
    return _cspimpl._c_api_output_adapter(mgr_capsule, engine, (int, capsule))


# Managed adapter definitions - note the manager_type argument
_managed_input_def = input_adapter_def(
    "my_managed_input",
    _create_managed_input,
    ts["T"],
    MyAdapterManager,  # <-- manager type
    typ="T",
    interval_ms=int,
    push_group=(object, None),  # Accept push_group kwarg
)

_managed_output_def = output_adapter_def(
    "my_managed_output",
    _create_managed_output,
    MyAdapterManager,  # <-- manager type
    input=ts["T"],
    prefix=str,
)
```

#### Use Managed Adapters in Your Graph

```python
import csp
from datetime import datetime, timedelta
from my_adapter import MyAdapterManager

@csp.graph
def my_graph():
    # Create a manager - all adapters share this instance
    mgr = MyAdapterManager(prefix="[MyApp] ")

    # Subscribe and publish through the manager
    data = mgr.subscribe(int, interval_ms=100)
    mgr.publish(data)

csp.run(my_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10))
```

#### Use in Your Graph

```python
import csp
from datetime import datetime, timedelta
from my_adapter import my_input, LogAdapter

@csp.graph
def my_graph():
    data = my_input(int, interval_ms=100)
    LogAdapter(data, prefix="[MyApp] ")

csp.run(my_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10))
```

## See Also

- [C API Reference](../api-references/C-APIs.md) - Complete API documentation
- [Write Output Adapters](Write-Output-Adapters.md) - Python output adapters
- [Write Realtime Input Adapters](Write-Realtime-Input-Adapters.md) - Python input adapters
- [C API Adapter Example](../../../examples/05_cpp/4_c_api_adapter/) - Working example
