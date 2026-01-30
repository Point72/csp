# CSP Engine Architecture

This document provides an in-depth technical overview of CSP's C++ engine implementation. It is intended for developers who want to understand or contribute to the core engine code.

For conceptual introductions to CSP graphs and nodes, see the [concepts documentation](../concepts/).

## Table of Contents

- [Overview](#overview)
- [Engine Lifecycle](#engine-lifecycle)
- [Core Components](#core-components)
  - [RootEngine](#rootengine)
  - [SRMWLockFreeQueue](#srmwlockfreequeue)
  - [CycleStepTable](#cyclesteptable)
  - [Scheduler](#scheduler)
  - [QueueWaiter and FdWaiter](#queuewaiter-and-fdwaiter)
- [Execution Model](#execution-model)
  - [Engine Cycles](#engine-cycles)
  - [Push Event Processing](#push-event-processing)
  - [Rank-Based Execution](#rank-based-execution)
- [Adapters](#adapters)
  - [InputAdapter](#inputadapter)
  - [OutputAdapter](#outputadapter)
  - [PushInputAdapter](#pushinputadapter)
  - [PushGroup Synchronization](#pushgroup-synchronization)
- [Asyncio Integration](#asyncio-integration)
  - [Decomposed Execution API](#decomposed-execution-api)
  - [FdWaiter for Native Event Loop Integration](#fdwaiter-for-native-event-loop-integration)
- [Thread Safety](#thread-safety)
- [Memory Management](#memory-management)

______________________________________________________________________

## Overview

CSP's engine is implemented in C++ for performance, with Python bindings via pybind11. The engine processes events in discrete **cycles**, where each cycle represents a single point in time. Within a cycle, nodes execute in **rank order** to ensure deterministic behavior.

Key design principles:

1. **Determinism**: Given the same inputs and same starttime, a CSP graph produces identical outputs
1. **Push-based events**: External data enters through lock-free queues, avoiding blocking the main engine thread
1. **Single-threaded execution**: The engine itself runs on a single thread; external threads push events that are processed in the next cycle
1. **Time-ordered processing**: Events are processed strictly in time order, with cycles executing at each unique timestamp

______________________________________________________________________

## Engine Lifecycle

The engine exposes a **decomposed execution API** that allows external event loops (like Python's asyncio) to drive execution:

```
┌─────────────────┐
│     start()     │  Initialize engine with starttime/endtime
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│processOneCycle()│  Execute one engine cycle (repeatable)
└────────┬────────┘
         │ (repeat)
         ▼
┌─────────────────┐
│    finish()     │  Clean up and return outputs
└─────────────────┘
```

### start(starttime, endtime)

Initializes the engine:

- Sets up the scheduler with the time range
- Prepares the cycle step table
- Initializes signal handlers for graceful shutdown
- Calls `start()` on all adapters

### processOneCycle(maxWait)

Executes a single engine cycle:

1. In realtime mode, waits up to `maxWait` seconds for push events or scheduled callbacks
1. Processes all pending push events from the lock-free queue
1. Executes all scheduled callbacks due at the current time
1. Runs the cycle step table (executes nodes by rank)
1. Returns `true` if the engine should continue, `false` when done

### finish()

Shuts down the engine:

- Calls `stop()` on all adapters
- Collects graph outputs
- Cleans up resources

This decomposition allows the `CspEventLoop` to interleave CSP execution with Python asyncio operations.

______________________________________________________________________

## Core Components

### RootEngine

**Header**: `cpp/csp/engine/RootEngine.h`

The `RootEngine` is the top-level orchestrator. Key members:

| Member             | Type                           | Purpose                                           |
| ------------------ | ------------------------------ | ------------------------------------------------- |
| `m_cycleStepTable` | `CycleStepTable`               | Executes consumers by rank                        |
| `m_scheduler`      | `Scheduler`                    | Time-based callback scheduling                    |
| `m_pushEventQueue` | `SRMWLockFreeQueue<PushEvent>` | Thread-safe external event queue                  |
| `m_queueWaiter`    | `QueueWaiter`                  | Blocks until events arrive (condition variable)   |
| `m_fdWaiter`       | `FdWaiter`                     | File descriptor for native event loop integration |
| `m_cycleCount`     | `uint64_t`                     | Monotonically increasing cycle counter            |

The engine maintains the current time via `now()` which returns the timestamp of the current cycle.

### SRMWLockFreeQueue

**Header**: `cpp/csp/core/SRMWLockFreeQueue.h`

A **S**ingle-**R**eader **M**ulti-**W**riter lock-free queue using atomic compare-exchange operations.

**Design**:

```
  Writer Threads                    Reader Thread
  ┌──────────┐                     ┌─────────────┐
  │ push(A)  │──┐                  │             │
  └──────────┘  │    ┌──────┐     │ popAll()    │
                ├──▶ │ HEAD │ ──▶ │ returns     │
  ┌──────────┐  │    └──────┘     │ linked list │
  │ push(B)  │──┘                  │             │
  └──────────┘                     └─────────────┘
```

**Key operations**:

- **`push(T)`**: Writers atomically prepend to a linked list head. Uses `compare_exchange_weak` in a loop.
- **`popAll()`**: Reader atomically swaps the head to null and returns the entire list. Reverses it to maintain FIFO order.
- **`Batch`**: Groups multiple pushes into a single atomic operation, ensuring related events stay together.

**Example usage in CSP**:

```cpp
// External thread pushing an event
m_pushEventQueue.push(new TypedPushEvent<int>(adapter, 42));
m_queueWaiter.notify();  // Wake up engine

// Engine thread consuming events
auto events = m_pushEventQueue.popAll();
for (auto& event : events) {
    event->invoke();
}
```

### CycleStepTable

**Header**: `cpp/csp/engine/CycleStepTable.h`

Manages **rank-based execution** of consumers (nodes and output adapters) within a cycle.

**Concept**: Each consumer has a **rank** determined by its position in the graph topology. Lower ranks execute first, ensuring that upstream nodes tick before downstream consumers see the data.

**Implementation**:

```
┌─────────────────────────────────────────┐
│ DynamicBitSet: dirty ranks              │
│ ┌───┬───┬───┬───┬───┬───┬───┬───┐      │
│ │ 0 │ 1 │ 0 │ 1 │ 0 │ 0 │ 1 │ 0 │ ...  │
│ └───┴───┴───┴───┴───┴───┴───┴───┘      │
│                                         │
│ m_table[rank] = vector<Consumer*>       │
│ rank 1: [NodeA, NodeB]                  │
│ rank 3: [NodeC]                         │
│ rank 6: [OutputAdapterX]                │
└─────────────────────────────────────────┘
```

When a consumer's input ticks:

1. The consumer's rank is marked dirty in the bitset
1. During cycle execution, the table iterates through dirty ranks in order
1. All consumers at each rank execute via `Consumer::execute()`
1. Consumers may produce outputs, marking downstream ranks as dirty

### Scheduler

**Header**: `cpp/csp/engine/Scheduler.h`

Handles time-based scheduling of callbacks (alarms, timers).

**Data structures**:

- **`EventMap`**: A map from `DateTime` to a list of `Event` objects
- **`Handle`**: An opaque reference to a scheduled event (used for cancellation)
- **`DynamicEngineStart` monitor**: Detects adapters added during runtime (for dynamic graphs)

**Key operations**:

```cpp
// Schedule a callback for a specific time
Handle scheduleCallback(DateTime when, Callback cb);

// Reschedule an existing callback
void rescheduleCallback(Handle h, DateTime newTime);

// Cancel a scheduled callback
void cancelCallback(Handle h);

// Get the next scheduled time
DateTime getNextTime();

// Execute all events at or before the given time
void executeEventsUpTo(DateTime time);
```

### QueueWaiter and FdWaiter

**Header**: `cpp/csp/core/QueueWaiter.h`

**QueueWaiter**: Traditional condition-variable-based waiting for push events.

```cpp
// Writer thread
void notify() {
    std::lock_guard<std::mutex> guard(m_lock);
    m_eventsPending = true;
    m_condition.notify_one();
}

// Engine thread
bool wait(TimeDelta maxWait) {
    std::unique_lock<std::mutex> lock(m_lock);
    m_condition.wait_for(lock, maxWait, [&]{ return m_eventsPending; });
    bool had_events = m_eventsPending;
    m_eventsPending = false;
    return had_events;
}
```

**FdWaiter**: File-descriptor-based signaling for native event loop integration. This allows external event loops (like asyncio) to use `select()`/`poll()`/`epoll()` to wait for CSP events.

Platform-specific implementations:

| Platform | Mechanism   | Description               |
| -------- | ----------- | ------------------------- |
| Linux    | `eventfd`   | Single fd, most efficient |
| macOS    | `pipe`      | Pair of fds (read/write)  |
| Windows  | Socket pair | Localhost TCP connection  |

```cpp
// Get fd for registration with external selector
int fd = fdWaiter.readFd();
selector.register(fd, EVENT_READ);

// When events available, writer calls:
fdWaiter.notify();

// Event loop detects fd readable, then:
fdWaiter.clear();  // Reset for next notification
engine.processOneCycle(0);  // Process events
```

______________________________________________________________________

## Execution Model

### Engine Cycles

Each cycle processes events at a single timestamp:

```
┌─────────────────────────────────────────────────────────────┐
│                    Engine Cycle N                           │
│                    Time: 2024-01-01 10:00:00.123            │
├─────────────────────────────────────────────────────────────┤
│ 1. Process push events from queue                           │
│    - External data pushed by adapter threads                │
│    - Each tick schedules consumer in CycleStepTable         │
│                                                             │
│ 2. Execute scheduler callbacks                              │
│    - Timer ticks, alarm callbacks                           │
│    - May schedule more consumers                            │
│                                                             │
│ 3. Execute CycleStepTable                                   │
│    - Iterate dirty ranks in order                           │
│    - Execute all consumers at each rank                     │
│    - Consumers may output, dirtying downstream ranks        │
│                                                             │
│ 4. Increment cycleCount                                     │
└─────────────────────────────────────────────────────────────┘
```

Between cycles, the engine either:

- **Simulation mode**: Jumps instantly to the next scheduled event time
- **Realtime mode**: Waits until wall-clock time reaches the next event, or push events arrive

### Push Event Processing

When external data arrives:

```
External Thread              Engine Thread
     │                            │
     │  push(event)               │
     │  ───────────▶              │
     │  notify()                  │
     │  ─────────────────────▶    │
     │                            │ (wake from wait)
     │                        popAll()
     │                            │
     │                        for each event:
     │                          adapter.consumeEvent()
     │                            │
     │                        cycleStepTable.execute()
```

### Rank-Based Execution

Rank assignment follows graph topology:

```
Input Adapters: rank 0
       │
       ▼
    ┌──────┐
    │Node A│ rank 1
    └──┬───┘
       │
   ┌───┴───┐
   ▼       ▼
┌──────┐ ┌──────┐
│Node B│ │Node C│ rank 2
└──┬───┘ └──┬───┘
   │       │
   └───┬───┘
       ▼
  ┌─────────┐
  │ Node D  │ rank 3
  └────┬────┘
       │
       ▼
┌───────────────┐
│Output Adapter │ rank 4
└───────────────┘
```

This ensures that when Node A ticks, Node B and C see A's output in the same cycle, and Node D sees B and C's outputs.

______________________________________________________________________

## Adapters

### InputAdapter

**Header**: `cpp/csp/engine/InputAdapter.h`

Base class for adapters that bring data into the graph. Inherits from `TimeSeriesProvider` (it *is* a time series).

**Key methods**:

- `start(DateTime, DateTime)`: Called when engine starts
- `stop()`: Called when engine stops
- `consumeTick<T>(value)`: Process an incoming value based on `PushMode`

**PushMode handling**:

| Mode             | Behavior                                           |
| ---------------- | -------------------------------------------------- |
| `LAST_VALUE`     | Multiple ticks at same time collapse to last value |
| `NON_COLLAPSING` | Each tick gets its own cycle (same timestamp)      |
| `BURST`          | All ticks at same time grouped into a vector       |

### OutputAdapter

**Header**: `cpp/csp/engine/OutputAdapter.h`

Base class for adapters that send data out of the graph. Inherits from `Consumer` (it listens to a time series).

**Key methods**:

- `link(TimeSeriesProvider*)`: Connect to input time series
- `executeImpl()`: Called when input ticks (override in subclasses)

### PushInputAdapter

**Header**: `cpp/csp/engine/PushInputAdapter.h`

Extends `InputAdapter` for thread-safe pushing from external threads.

**Key features**:

- `pushTick<T>(value)`: Thread-safe push via `SRMWLockFreeQueue`
- Integrates with `PushGroup` for synchronization across multiple adapters

```cpp
template<typename T>
void PushInputAdapter::pushTick(const T& value) {
    auto event = new TypedPushEvent<T>(this, value);
    if (m_pushGroup)
        m_pushGroup->push(event);
    else {
        m_pushEventQueue.push(event);
        m_queueWaiter.notify();
    }
}
```

### PushGroup Synchronization

When multiple adapters need to push related events atomically:

```cpp
// PushGroup states:
enum State { NONE, LOCKING, LOCKED };

// Usage pattern:
pushGroup.startBatch();          // Acquire lock
adapter1.pushTick(value1);       // Events buffered
adapter2.pushTick(value2);
pushGroup.endBatch();            // Atomically push all events
```

This ensures related events from different adapters arrive in the same engine cycle.

______________________________________________________________________

## Asyncio Integration

### Decomposed Execution API

The engine exposes a Python-accessible decomposed API via `PyEngine`:

```python
engine = _cspimpl.PyEngine(realtime=True)

# Initialize
engine.start(starttime, endtime)

# Run cycles manually
while engine.process_one_cycle(max_wait=0.1):
    # Can do asyncio work here between cycles
    await asyncio.sleep(0)

# Cleanup
results = engine.finish()
```

This allows `CspEventLoop` to integrate CSP cycles with Python's asyncio.

### FdWaiter for Native Event Loop Integration

The engine provides a wakeup file descriptor:

```python
# Get the fd for selector registration
fd = engine.get_wakeup_fd()
selector.register(fd, selectors.EVENT_READ)

# When fd becomes readable, CSP has events
events = selector.select(timeout)
for key, mask in events:
    if key.fd == fd:
        engine.clear_wakeup_fd()
        engine.process_one_cycle(0)
```

Benefits:

- No polling required
- Native integration with asyncio's event loop
- Efficient wakeup via OS-level fd signaling

______________________________________________________________________

## Thread Safety

CSP uses a **single-threaded engine with thread-safe ingestion**:

| Operation                          | Thread Safety                 |
| ---------------------------------- | ----------------------------- |
| `pushTick()` on adapters           | Thread-safe (lock-free queue) |
| `processOneCycle()`                | Single-threaded only          |
| `scheduleCallback()` within engine | Single-threaded only          |
| `SRMWLockFreeQueue.push()`         | Thread-safe                   |
| `SRMWLockFreeQueue.popAll()`       | Single reader only            |

**Rules for external code**:

1. Push events via `PushInputAdapter` from any thread
1. Never call engine methods from external threads (except push)
1. Use `PushGroup` when multiple adapters need atomic batching

______________________________________________________________________

## Memory Management

CSP uses **engine-owned** and **reference-counted** patterns:

### EngineOwned

Objects that inherit from `EngineOwned` are tracked by the engine and cleaned up on shutdown:

```cpp
class InputAdapter : public TimeSeriesProvider, public EngineOwned {
    // Engine deletes this on shutdown
};
```

### Push Events

Push events are allocated by writers and deleted after processing:

```cpp
// Writer allocates
auto event = new TypedPushEvent<int>(adapter, 42);
queue.push(event);

// Engine consumes and deletes
auto events = queue.popAll();
for (auto& event : events) {
    event->invoke();
    delete event;
}
```

### Time Series Buffers

Time series data uses ring buffers with configurable history:

```python
# Python API
csp.set_buffering_policy(ts, tick_count=100)
csp.set_buffering_policy(ts, tick_history=timedelta(minutes=5))
```

______________________________________________________________________

## Further Reading

- [CSP Node concepts](../concepts/CSP-Node.md) - Node anatomy and lifecycle hooks
- [Execution Modes](../concepts/Execution-Modes.md) - Simulation vs realtime
- [Adapters](../concepts/Adapters.md) - Writing custom adapters
- [Asyncio Integration](../how-tos/Asyncio-Integration.md) - Using CSP with asyncio
