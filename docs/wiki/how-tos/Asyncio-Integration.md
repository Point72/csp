# CSP Event Loop Integration

CSP provides three complementary integration patterns with Python's `asyncio` framework:

1. **Same-Thread Asyncio Mode (Default)**: In realtime mode, CSP runs async operations on the same thread by default
1. **Standalone Event Loop**: Use CSP as the asyncio event loop backend (similar to uvloop)
1. **Bridge with Running Graph**: Interleave asyncio operations with a running CSP graph

## Overview

The `csp.run()` function in **realtime mode** automatically runs async operations on the same thread as CSP:

- **`asyncio_on_thread=False`** (default in realtime): Same-thread asyncio execution, eliminating cross-thread synchronization overhead
- **`asyncio_on_thread=True`**: Run async operations on a background thread (legacy behavior)

The `csp.event_loop` module provides:

**Standalone Event Loop:**

- **`CspEventLoop`**: An asyncio-compatible event loop backed by CSP's scheduler
- **`CspEventLoopPolicy`**: An event loop policy for using CSP loops with asyncio
- **`run()`**: A convenience function for running coroutines with CSP's event loop
- **`new_event_loop()`**: Factory function to create a new CSP event loop

**Bridge with Running Graph:**

- **`AsyncioBridge`**: Bridge for pushing data from asyncio to CSP graphs
- **`BidirectionalBridge`**: Bridge supporting two-way communication

## Part 0: Same-Thread Asyncio Mode (Default)

When running CSP graphs in realtime mode, async operations run on the same thread by default:

```python
import csp
from datetime import timedelta
from typing import AsyncIterator
import asyncio

async def fetch_data(count: int) -> AsyncIterator[int]:
    """Async generator that fetches data."""
    for i in range(count):
        await asyncio.sleep(0.1)
        yield i * 10

@csp.graph
def my_graph():
    # Async adapters automatically use CSP's asyncio loop
    data = csp.async_for(fetch_data(5))
    csp.print("data", data)

# In realtime mode, asyncio runs on the same thread by default
csp.run(my_graph, realtime=True, endtime=timedelta(seconds=2))
```

### How It Works

When you call `csp.run(..., realtime=True)`:

1. CSP creates a new asyncio event loop on the **current thread**
1. The CSP engine yields control to the asyncio loop between engine cycles, allowing async tasks to run
1. All async adapters (`async_for`, `async_in`, `async_out`, etc.) automatically detect this mode and run on the **same thread** as CSP
1. No background thread is needed - this eliminates cross-thread synchronization overhead

**To use the legacy background thread mode**, set `asyncio_on_thread=True`:

```python
# Explicitly use background thread for async operations
csp.run(my_graph, realtime=True, endtime=timedelta(seconds=2), asyncio_on_thread=True)
```

### Detecting Asyncio Mode

You can check if you're running in CSP's asyncio mode:

```python
from csp.impl.async_adapter import is_csp_asyncio_mode, get_csp_asyncio_loop

@csp.node
def my_node(x: ts[int]) -> ts[int]:
    if csp.ticked(x):
        if is_csp_asyncio_mode():
            loop = get_csp_asyncio_loop()
            # We're running in asyncio mode, can schedule directly on the loop
        return x * 2
```

### Requirements

- Same-thread asyncio mode is only active in realtime mode (`realtime=True`)
- The graph runs synchronously (blocks until complete)
- The return value is the same as normal `csp.run()`

## Part 1: Standalone Event Loop

### Basic Usage

The simplest way to use CSP's asyncio integration is through the `run()` function:

```python
import csp.event_loop as csp_event_loop

async def main():
    print("Hello from CSP asyncio!")
    await asyncio.sleep(1)
    return "done"

result = csp_event_loop.run(main())
print(result)  # "done"
```

### Using as Event Loop Policy

You can set CSP as the default event loop for all asyncio operations:

```python
import asyncio
import csp.event_loop as csp_event_loop

# Set CSP as the event loop policy
asyncio.set_event_loop_policy(csp_event_loop.EventLoopPolicy())

# Now all asyncio operations use CSP's event loop
async def main():
    await asyncio.sleep(0.1)
    return "Hello!"

# This will use CSP's event loop
result = asyncio.run(main())
```

### Creating Event Loops Manually

For more control, you can create and manage event loops directly:

```python
from csp.event_loop import new_event_loop

loop = new_event_loop()
try:
    result = loop.run_until_complete(my_coroutine())
finally:
    loop.close()
```

## Part 2: Bridge with Running CSP Graph

The bridge integration allows asyncio operations to interact with a running CSP graph.
This is useful when you want to:

- Push data from asyncio callbacks or coroutines into CSP
- Schedule callbacks using `call_later` and `call_at` that feed data to CSP
- Coordinate asyncio timing with CSP's engine time (`csp.now()`)
- Enable bidirectional communication between asyncio and CSP nodes

### Quick Start with AsyncioBridge

```python
import csp
from csp.event_loop import AsyncioBridge
from csp.utils.datetime import utc_now
from datetime import timedelta

# Create the bridge
bridge = AsyncioBridge(int, "data_feed")

@csp.node
def process(data: csp.ts[int]) -> csp.ts[str]:
    if csp.ticked(data):
        return f"Received {data} at {csp.now()}"

@csp.graph
def my_graph():
    # Wire the bridge's adapter into the graph
    data = bridge.adapter.out()
    result = process(data)
    csp.add_graph_output("result", result)

# Start the bridge
start_time = utc_now()
bridge.start(start_time)

# Run the CSP graph in a thread
runner = csp.run_on_thread(
    my_graph,
    realtime=True,
    starttime=start_time,
    endtime=timedelta(seconds=5)
)

# Wait for adapter to be ready
bridge.wait_for_adapter(timeout=1.0)

# Push data from asyncio
bridge.call_later(0.5, lambda: bridge.push(1))
bridge.call_later(1.0, lambda: bridge.push(2))
bridge.call_later(1.5, lambda: bridge.push(3))

# Wait for completion
results = runner.join()
bridge.stop()

print(results["result"])
```

### Scheduling with call_later and call_at

The bridge provides asyncio-style scheduling methods:

```python
from datetime import datetime, timedelta

bridge = AsyncioBridge(str, "events")
bridge.start()

# Schedule callback after delay
bridge.call_later(1.0, lambda: bridge.push("after 1 second"))

# Schedule at specific datetime
target_time = datetime.utcnow() + timedelta(seconds=2)
bridge.call_at(target_time, lambda: bridge.push("at specific time"))

# Schedule at offset from start time (aligned with CSP time)
bridge.call_at_offset(
    timedelta(milliseconds=500),
    lambda: bridge.push("at 500ms from start")
)
```

### Running Async Coroutines

You can run full asyncio coroutines that interact with CSP:

```python
async def fetch_data_and_push():
    """Coroutine that fetches data and pushes to CSP."""
    for i in range(5):
        await asyncio.sleep(0.2)
        # Simulate fetching data
        data = {"value": i, "timestamp": time.time()}
        bridge.push(data)

# Run the coroutine
future = bridge.run_coroutine(fetch_data_and_push())

# Optionally wait for completion
future.result(timeout=10.0)
```

### Coordinating with CSP Time (csp.now())

When scheduling callbacks, you can align with CSP's engine start time:

```python
@csp.node
def log_with_time(data: csp.ts[str]) -> csp.ts[str]:
    if csp.ticked(data):
        # csp.now() shows the engine time
        return f"[{csp.now()}] {data}"

# The bridge uses wall-clock time, but you can align with CSP start
start_time = utc_now()
bridge.start(start_time)

# This callback fires at start_time + 1 second
# which aligns with csp.now() being approximately 1 second into the run
bridge.call_at_offset(timedelta(seconds=1), lambda: bridge.push("1s mark"))
```

### Bidirectional Communication

For two-way communication, use `BidirectionalBridge`:

```python
from csp.event_loop import BidirectionalBridge

bridge = BidirectionalBridge(str, "bidi")

@csp.node
def process_and_respond(data: csp.ts[str], bridge_ref: object) -> csp.ts[str]:
    if csp.ticked(data):
        response = f"processed: {data}"
        # Emit back to asyncio
        bridge_ref.emit({"input": data, "output": response})
        return response

@csp.graph
def my_graph():
    data = bridge.adapter.out()
    result = process_and_respond(data, bridge)
    csp.add_graph_output("results", result)

# Register callback to receive events from CSP
def on_csp_event(event):
    print(f"Received from CSP: {event}")

bridge.on_event(on_csp_event)

# Start everything
bridge.start()
runner = csp.run_on_thread(my_graph, realtime=True, ...)

# Push to CSP
bridge.push("hello")

# Later, the on_csp_event callback receives:
# {"input": "hello", "output": "processed: hello"}
```

### Combining CSP Timers with Async Callbacks

You can run CSP's internal timers alongside async callbacks:

```python
@csp.node
def combine(timer: csp.ts[bool], async_data: csp.ts[str]) -> csp.ts[str]:
    if csp.ticked(timer):
        return "timer tick"
    if csp.ticked(async_data):
        return f"async: {async_data}"

@csp.graph
def my_graph():
    # CSP's timer fires every 100ms
    timer = csp.timer(timedelta(milliseconds=100))
    # Async data comes from the bridge
    async_data = bridge.adapter.out()

    result = combine(timer, async_data)
    csp.add_graph_output("events", result)

# Schedule async callbacks at different intervals
bridge.call_later(0.05, lambda: bridge.push("early"))
bridge.call_later(0.15, lambda: bridge.push("middle"))
bridge.call_later(0.25, lambda: bridge.push("late"))

# The graph receives interleaved timer and async events
```

## Features

### Full Asyncio Compatibility

The CSP event loop is compatible with standard asyncio primitives:

```python
import asyncio
import csp.event_loop as csp_event_loop

async def producer(queue):
    for i in range(5):
        await asyncio.sleep(0.1)
        await queue.put(i)
    await queue.put(None)  # Signal end

async def consumer(queue):
    results = []
    while True:
        item = await queue.get()
        if item is None:
            break
        results.append(item)
    return results

async def main():
    queue = asyncio.Queue()

    # Run producer and consumer concurrently
    producer_task = asyncio.create_task(producer(queue))
    consumer_task = asyncio.create_task(consumer(queue))

    await producer_task
    results = await consumer_task

    return results

results = csp_event_loop.run(main())
print(results)  # [0, 1, 2, 3, 4]
```

### Synchronization Primitives

All asyncio synchronization primitives work with CSP's event loop:

```python
import asyncio
import csp.event_loop as csp_event_loop

async def main():
    # Locks
    lock = asyncio.Lock()
    async with lock:
        print("Holding lock")

    # Events
    event = asyncio.Event()
    event.set()
    await event.wait()

    # Semaphores
    sem = asyncio.Semaphore(3)
    async with sem:
        print("Acquired semaphore")

    # Conditions
    condition = asyncio.Condition()
    async with condition:
        condition.notify_all()

csp_event_loop.run(main())
```

### Concurrent Operations

Use `asyncio.gather()`, `asyncio.wait()`, and other concurrent operations:

```python
import asyncio
import csp.event_loop as csp_event_loop

async def fetch_data(url):
    await asyncio.sleep(0.1)  # Simulate network delay
    return f"data from {url}"

async def main():
    urls = ["url1", "url2", "url3"]

    # Gather results concurrently
    results = await asyncio.gather(*[fetch_data(url) for url in urls])
    return results

results = csp_event_loop.run(main())
```

### Timeouts

Use `asyncio.wait_for()` for timeout operations:

```python
import asyncio
import csp.event_loop as csp_event_loop

async def slow_operation():
    await asyncio.sleep(10)
    return "done"

async def main():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=1.0)
    except asyncio.TimeoutError:
        result = "timeout"
    return result

result = csp_event_loop.run(main())  # "timeout"
```

### Thread Pool Executor

Run blocking operations in a thread pool:

```python
import asyncio
import csp.event_loop as csp_event_loop
import time

def blocking_io():
    time.sleep(0.1)
    return "data"

async def main():
    loop = asyncio.get_running_loop()

    # Run in default executor
    result = await loop.run_in_executor(None, blocking_io)
    return result

result = csp_event_loop.run(main())
```

### I/O Operations

Socket and file descriptor operations:

```python
import asyncio
import socket
import csp.event_loop as csp_event_loop

async def main():
    loop = asyncio.get_running_loop()

    # Create a socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setblocking(False)

    # Connect asynchronously
    try:
        await loop.sock_connect(sock, ('example.com', 80))

        # Send data
        await loop.sock_sendall(sock, b'GET / HTTP/1.0\r\n\r\n')

        # Receive data
        data = await loop.sock_recv(sock, 1024)
        return data[:50]
    finally:
        sock.close()

# result = csp_event_loop.run(main())
```

### Callback Scheduling

Schedule callbacks directly on the event loop:

```python
import csp.event_loop as csp_event_loop

loop = csp_event_loop.new_event_loop()

results = []

def my_callback(value):
    results.append(value)

# Schedule callbacks
loop.call_soon(my_callback, "immediate")
loop.call_later(0.1, my_callback, "delayed")
loop.call_soon(loop.stop)

loop.run_forever()
loop.close()

print(results)  # ["immediate"]
```

### Exception Handling

Custom exception handlers:

```python
import asyncio
import csp.event_loop as csp_event_loop

def my_exception_handler(loop, context):
    exception = context.get("exception")
    message = context.get("message")
    print(f"Caught exception: {exception}, message: {message}")

async def main():
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(my_exception_handler)

    # This exception will be caught by our handler
    async def bad_task():
        raise ValueError("oops")

    task = asyncio.create_task(bad_task())
    await asyncio.sleep(0.1)  # Let the task run

csp_event_loop.run(main())
```

### Debug Mode

Enable debug mode for development:

```python
import csp.event_loop as csp_event_loop

async def main():
    loop = asyncio.get_running_loop()
    print(f"Debug mode: {loop.get_debug()}")

# Enable debug mode
csp_event_loop.run(main(), debug=True)

# Or set via environment variable
# PYTHONASYNCIODEBUG=1 python script.py
```

## API Reference

### `csp.event_loop.run(main, *, loop_factory=None, debug=None)`

Run a coroutine using the CSP event loop.

**Parameters:**

- `main`: The coroutine to run
- `loop_factory`: Optional factory function to create the event loop. Defaults to `new_event_loop`.
- `debug`: If True, run in debug mode.

**Returns:** The result of the coroutine.

### `csp.event_loop.new_event_loop()`

Create and return a new CSP event loop.

**Returns:** A new `CspEventLoop` instance.

### `csp.event_loop.CspEventLoop`

An asyncio-compatible event loop backed by CSP's scheduler.

The loop implements the full `asyncio.AbstractEventLoop` interface, including:

- `run_until_complete(future)`: Run until a future completes
- `run_forever()`: Run until `stop()` is called
- `stop()`: Stop the loop
- `close()`: Close the loop
- `is_running()`: Check if the loop is running
- `is_closed()`: Check if the loop is closed
- `call_soon(callback, *args)`: Schedule a callback
- `call_later(delay, callback, *args)`: Schedule a delayed callback
- `call_at(when, callback, *args)`: Schedule a callback at absolute time
- `call_soon_threadsafe(callback, *args)`: Thread-safe callback scheduling
- `create_future()`: Create a new Future
- `create_task(coro)`: Create a new Task
- `run_in_executor(executor, func, *args)`: Run in thread pool
- `add_reader(fd, callback, *args)`: Add file descriptor reader
- `remove_reader(fd)`: Remove file descriptor reader
- `add_writer(fd, callback, *args)`: Add file descriptor writer
- `remove_writer(fd)`: Remove file descriptor writer
- `time()`: Get current loop time
- `get_debug()`: Get debug mode status
- `set_debug(enabled)`: Set debug mode

### `csp.event_loop.CspEventLoopPolicy`

Event loop policy for CSP-backed asyncio.

Methods:

- `get_event_loop()`: Get the event loop for the current context
- `set_event_loop(loop)`: Set the event loop for the current context
- `new_event_loop()`: Create a new event loop

Alias: `csp.event_loop.EventLoopPolicy`

### `csp.event_loop.AsyncioBridge`

Bridge between asyncio and running CSP graphs.

**Constructor:**

```python
AsyncioBridge(adapter_type: type = object, name: str = "asyncio_bridge")
```

**Parameters:**

- `adapter_type`: The type of data to push through the adapter
- `name`: Name for the push adapter (for debugging)

**Properties:**

- `adapter`: The `GenericPushAdapter` to wire into your CSP graph
- `is_running`: Whether the bridge is currently running
- `loop`: The underlying asyncio event loop (if started)

**Methods:**

- `start(start_time=None)`: Start the asyncio event loop in a background thread
- `stop(timeout=5.0)`: Stop the asyncio event loop
- `push(value)`: Push a value to the CSP graph
- `call_soon(callback, *args)`: Schedule a callback immediately
- `call_later(delay, callback, *args)`: Schedule a callback after delay seconds
- `call_at(when, callback, *args)`: Schedule a callback at a specific datetime
- `call_at_offset(offset, callback, *args)`: Schedule at offset from start time
- `run_coroutine(coro)`: Run an asyncio coroutine
- `wait_for_adapter(timeout=None)`: Wait for adapter to be bound to graph
- `time()`: Get current time in seconds since epoch
- `elapsed_since_start()`: Get time elapsed since start

### `csp.event_loop.BidirectionalBridge`

Extended bridge supporting two-way communication.

Inherits all methods from `AsyncioBridge`, plus:

**Additional Methods:**

- `on_event(callback)`: Register a callback to receive events from CSP
- `off_event(callback)`: Unregister an event callback
- `emit(value)`: Emit an event from CSP to asyncio (call from CSP nodes)

## Best Practices

### 1. Use `run()` for simple scripts

For simple scripts and applications, use `csp.event_loop.run()`:

```python
import csp.event_loop as csp_event_loop

async def main():
    # Your async code here
    pass

csp_event_loop.run(main())
```

### 2. Always close loops and stop bridges

When creating loops manually, always close them:

```python
loop = csp_event_loop.new_event_loop()
try:
    loop.run_until_complete(main())
finally:
    loop.close()
```

When using bridges, always stop them:

```python
bridge = AsyncioBridge(int, "data")
bridge.start()
try:
    # ... run your graph ...
finally:
    bridge.stop()
```

### 3. Handle shutdown gracefully

Shutdown async generators and executors:

```python
async def shutdown(loop):
    await loop.shutdown_asyncgens()
    await loop.shutdown_default_executor()
```

### 4. Use context managers for resources

```python
async def main():
    async with aiofiles.open('file.txt') as f:
        content = await f.read()
```

### 5. Wait for adapter binding

When using the bridge with CSP, wait for the adapter to be ready:

```python
bridge.start()
runner = csp.run_on_thread(my_graph, realtime=True, ...)

# Wait for CSP graph to start and bind the adapter
bridge.wait_for_adapter(timeout=1.0)

# Now it's safe to push data
bridge.push(data)
```

## Limitations

The current implementation has some limitations:

1. **Subprocess support**: `subprocess_exec()` and `subprocess_shell()` are not yet implemented.

1. **SSL/TLS**: Direct SSL support requires additional implementation.

1. **Signal handlers**: Signal handling works but has some platform-specific limitations.

1. **Bridge timing**: The bridge uses wall-clock time, not CSP engine time. Use `call_at_offset` to align with CSP start time.

## Comparison: Choosing the Right Integration

| Feature            | `csp.run(realtime=True)`        | `CspEventLoop`                        | `AsyncioBridge`                |
| ------------------ | ------------------------------- | ------------------------------------- | ------------------------------ |
| **Use case**       | CSP graph with async operations | Pure asyncio code with CSP scheduling | CSP graph receiving async data |
| **Threading**      | Single-threaded (default)       | Single-threaded                       | Multi-threaded                 |
| **CSP graph**      | Yes (main focus)                | No (pure asyncio)                     | Yes (main focus)               |
| **Async adapters** | Automatic integration           | Need to run in the loop               | Need explicit bridge           |
| **Complexity**     | Low                             | Low                                   | Medium                         |
| **When to use**    | Async I/O in CSP nodes          | Replace asyncio.run()                 | Push external data to CSP      |

### Quick Decision Guide

1. **I have a CSP graph with async operations (fetch APIs, async I/O):**
   → Use `csp.run(my_graph, realtime=True, ...)` - same-thread asyncio is the default

1. **I want to use asyncio code and need CSP's scheduler:**
   → Use `CspEventLoop` or `csp.event_loop.run()`

1. **I have a running CSP graph and need to push data from external async sources:**
   → Use `AsyncioBridge` or `BidirectionalBridge`

1. **I want async adapters to work without extra configuration:**
   → Just use `realtime=True` - adapters auto-detect the asyncio mode

## See Also

- [Async Adapters Reference](Async.md) - Detailed async adapter documentation
- [Python asyncio documentation](https://docs.python.org/3/library/asyncio.html)
- [CSP Documentation](../README.md)
- [uvloop](https://github.com/MagicStack/uvloop) - Similar project for libuv-based event loop
- [Example: CSP Asyncio Integration](https://github.com/Point72/csp/tree/main/examples/06_advanced/e2_csp_event_loop_integration.py)
