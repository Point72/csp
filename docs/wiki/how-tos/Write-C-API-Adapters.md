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
    - [Create Python Wrapper](#create-python-wrapper)
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

To use your C adapter from Python, you need to create a Python extension module that wraps the C functions.

#### Create Python Bindings (C code)

```c
#include <Python.h>
#include <csp/python/c/PyOutputAdapter.h>
#include "my_adapter.h"

static PyObject* create_log_adapter_py(PyObject* self, PyObject* args)
{
    const char* prefix = NULL;
    if (!PyArg_ParseTuple(args, "|s", &prefix)) {
        return NULL;
    }

    CCspOutputAdapterVTable vtable = create_log_adapter(prefix);
    if (!vtable.execute) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create adapter");
        return NULL;
    }

    return ccsp_py_create_output_adapter_capsule(&vtable);
}

static PyMethodDef methods[] = {
    {"_log_adapter", create_log_adapter_py, METH_VARARGS, "Create log adapter"},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "my_adapter", NULL, -1, methods
};

PyMODINIT_FUNC PyInit_my_adapter(void) {
    return PyModule_Create(&module);
}
```

#### Create Python Wrapper

```python
# my_adapter.py
from csp import ts
from csp.impl.wiring import output_adapter_def
from csp.lib import my_adapter as _impl

LogAdapter = output_adapter_def(
    "LogAdapter",
    _impl._log_adapter,
    input=ts["T"],
    prefix=str,
)
```

#### Use in Your Graph

```python
import csp
from my_adapter import LogAdapter

@csp.graph
def my_graph():
    data = csp.timer(timedelta(seconds=1), "tick")
    LogAdapter(data, prefix="[MyApp] ")

csp.run(my_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10))
```

## See Also

- [C API Reference](../api-references/C-APIs.md) - Complete API documentation
- [Write Output Adapters](Write-Output-Adapters.md) - Python output adapters
- [Write Realtime Input Adapters](Write-Realtime-Input-Adapters.md) - Python input adapters
- [Example: C API Adapter](../../examples/04_writing_adapters/e8_c_api_adapter.py) - Working example
