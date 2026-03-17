## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Event Loop Options](#event-loop-options)
  - [Option 1: Default (Shared Background Loop)](#option-1-default-shared-background-loop)
  - [Option 2: CSP Asyncio Mode](#option-2-csp-asyncio-mode)
  - [Option 3: Custom Event Loop](#option-3-custom-event-loop)
  - [Choosing the Right Option](#choosing-the-right-option)
- [Graph-Level Async Adapters](#graph-level-async-adapters)
  - [async_for - Async Generator to Time Series](#async_for---async-generator-to-time-series)
  - [async_in - Single Async Value to Time Series](#async_in---single-async-value-to-time-series)
  - [async_out - Time Series to Async Function](#async_out---time-series-to-async-function)
  - [async_node - Transform Time Series via Async](#async_node---transform-time-series-via-async)
- [Node-Level Async Operations](#node-level-async-operations)
  - [await\_ - Blocking Await in Nodes](#await_---blocking-await-in-nodes)
  - [async_alarm - Alarm-Like Async Pattern](#async_alarm---alarm-like-async-pattern)
  - [AsyncContext - Persistent Async Event Loop (Advanced)](#asynccontext---persistent-async-event-loop-advanced)
- [Complete Example](#complete-example)
- [Best Practices](#best-practices)

## Introduction

CSP is fundamentally a synchronous, deterministic event processing framework. However, modern applications often need to interact with async APIs, external services, or I/O-bound operations that are naturally asynchronous.

The `csp.impl.async_adapter` module provides several tools to bridge Python's `asyncio` with CSP's synchronous graph processing:

| Function          | Purpose                                                    |
| ----------------- | ---------------------------------------------------------- |
| `csp.async_for`   | Convert an async generator to a CSP time series            |
| `csp.async_in`    | Convert a single async value to a time series (ticks once) |
| `csp.async_out`   | Invoke an async function on each tick (side effects)       |
| `csp.async_node`  | Transform time series values via an async function         |
| `csp.await_`      | Await async code inside a CSP node (blocking)              |
| `csp.async_alarm` | Alarm-like pattern for async operations in nodes           |

## Event Loop Options

All async adapters accept an optional `loop` parameter that controls which asyncio event loop is used for running async operations. CSP provides three ways to handle this:

### Option 1: Default (Shared Background Loop)

When you don't specify a loop, CSP automatically uses a shared background loop:

```python
@csp.graph
def my_graph():
    # Uses shared background loop (default)
    values = csp.async_for(my_async_gen(10))
    result = csp.async_in(fetch_config())
    csp.async_out(values, send_to_api)
```

This shared loop:

- Runs in a dedicated background thread
- Is lazily initialized on first use
- Is shared by all async adapters (efficient)
- Is automatically cleaned up on process exit

### Option 2: Same-Thread Asyncio Mode (Default in Realtime)

In realtime mode, CSP runs async operations on the same thread by default. This allows async operations to run directly on CSP's engine loop, eliminating the need for a separate background thread:

```python
@csp.graph
def my_graph():
    # All async adapters automatically use CSP's asyncio loop
    updates = csp.async_for(fetch_updates(10))
    config = csp.async_in(get_config())
    doubled = csp.async_node(updates, async_double)
    csp.print("doubled", doubled)

# In realtime mode, same-thread asyncio is the default
csp.run(my_graph, realtime=True, endtime=timedelta(seconds=5))
```

Key benefits:

- **Same-thread execution**: Async operations run on the same thread as the CSP engine
- **No threading overhead**: No cross-thread synchronization needed
- **Direct integration**: Async code can interact more naturally with CSP's scheduler
- **Automatic detection**: All async adapters detect CSP asyncio mode and use it automatically

To use the legacy background thread mode instead:

```python
# Explicitly run async on a background thread
csp.run(my_graph, realtime=True, endtime=timedelta(seconds=5), asyncio_on_thread=True)
```

To check if you're in asyncio mode:

```python
from csp.impl.async_adapter import is_csp_asyncio_mode, get_csp_asyncio_loop

if is_csp_asyncio_mode():
    loop = get_csp_asyncio_loop()
    print(f"Running in CSP asyncio mode on {loop}")
```

### Option 3: Custom Event Loop

You can provide your own event loop to any async adapter:

```python
import asyncio

# Create a custom loop
my_loop = asyncio.new_event_loop()

@csp.graph
def my_graph():
    # All operations use the custom loop
    values = csp.async_for(my_async_gen(10), loop=my_loop)
    result = csp.async_in(fetch_data(), loop=my_loop)
    csp.async_out(values, send_data, loop=my_loop)
    transformed = csp.async_node(values, transform, loop=my_loop)
```

### Choosing the Right Option

| Scenario                                    | Recommended Option                                 |
| ------------------------------------------- | -------------------------------------------------- |
| Simple async I/O, realtime mode             | **Default** (`realtime=True`, same-thread asyncio) |
| Need background thread for async            | **Background thread** (`asyncio_on_thread=True`)   |
| Need to coordinate with external async code | **Custom Loop** (pass `loop` parameter)            |
| Using CspEventLoop for asyncio integration  | **Default** (detects running loop automatically)   |

The async adapters use this priority for finding a loop:

1. If CSP is running in realtime mode with same-thread asyncio (the default), use that loop
1. If there's a running asyncio loop (e.g., CspEventLoop), use it
1. Otherwise, use the shared background loop

## Graph-Level Async Adapters

These adapters work at the graph level to bridge async code with CSP time series.

### async_for - Async Generator to Time Series

`csp.async_for` converts an async generator into a CSP time series. Each yielded value becomes a tick.

**Signature:**

```python
def async_for(
    async_gen_or_func: AsyncIterator[T],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts[T]
```

**Parameters:**

- `async_gen_or_func`: An async generator instance (result of calling an async generator function)
- `loop`: Event loop to use. If None, uses the best available loop (see [Event Loop Options](#event-loop-options))

**Example:**

```python
from typing import AsyncIterator
import asyncio
import csp
from datetime import timedelta

async def fetch_updates(count: int) -> AsyncIterator[dict]:
    """Async generator that yields updates from an external source."""
    for i in range(count):
        await asyncio.sleep(0.1)  # Simulate async I/O
        yield {"id": i, "value": i * 10}

@csp.graph
def my_graph():
    # Convert async generator to time series
    updates = csp.async_for(fetch_updates(10))
    csp.print("update", updates)

csp.run(my_graph, realtime=True, endtime=timedelta(seconds=2))
```

**Key points:**

- In realtime mode, the async generator runs on the same thread as CSP (or on a background thread if `asyncio_on_thread=True`)
- Type is inferred from the `AsyncIterator[T]` return annotation
- Runs until the generator is exhausted or the graph ends

### async_in - Single Async Value to Time Series

`csp.async_in` converts a single async coroutine result into a time series that ticks once.

**Signature:**

```python
def async_in(
    coro: Awaitable[T],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts[T]
```

**Parameters:**

- `coro`: A coroutine instance (result of calling an async function)
- `loop`: Event loop to use. If None, uses the best available loop (see [Event Loop Options](#event-loop-options))

**Example:**

```python
async def fetch_config() -> dict:
    """Fetch configuration from an async API."""
    await asyncio.sleep(0.1)
    return {"setting": "value"}

@csp.graph
def my_graph():
    # Single async value that ticks once when ready
    config = csp.async_in(fetch_config())
    csp.print("config", config)
```

**Key points:**

- Ticks exactly once when the coroutine completes
- Useful for initialization or one-time async fetches
- Type is inferred from the return annotation

### async_out - Time Series to Async Function

`csp.async_out` invokes an async function each time the input time series ticks. This is useful for async side effects like sending data to external services.

**Signature:**

```python
@csp.node
def async_out(
    x: ts[T],
    async_func: Callable[[T], Awaitable[None]],
    loop: Optional[asyncio.AbstractEventLoop] = None,
)
```

**Parameters:**

- `x`: Input time series that triggers the async function
- `async_func`: An async function that takes the ticked value and returns None
- `loop`: Event loop to use. If None, uses the best available loop (see [Event Loop Options](#event-loop-options))

**Example:**

```python
async def send_to_api(value: int) -> None:
    """Send value to an external async API."""
    await asyncio.sleep(0.1)  # Simulate async I/O
    print(f"Sent: {value}")

@csp.graph
def my_graph():
    counter = csp.count(csp.timer(timedelta(seconds=0.5), True))

    # Invoke async function on each tick
    csp.async_out(counter, send_to_api)
```

**Key points:**

- Each tick triggers an async operation
- Operations run concurrently on the event loop
- No return value (fire-and-forget for side effects)

### async_node - Transform Time Series via Async

`csp.async_node` transforms input time series values through an async function, producing an output time series.

**Signature:**

```python
def async_node(
    x: ts[T],
    async_func: Callable[[T], Awaitable[U]],
    *,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ts[U]
```

**Parameters:**

- `x`: Input time series
- `async_func`: An async function that transforms the input value
- `loop`: Event loop to use. If None, uses the best available loop (see [Event Loop Options](#event-loop-options))

**Example:**

```python
async def fetch_details(id: int) -> dict:
    """Fetch details for an ID from an async API."""
    await asyncio.sleep(0.1)
    return {"id": id, "details": f"Details for {id}"}

@csp.graph
def my_graph():
    ids = csp.count(csp.timer(timedelta(seconds=0.5), True))

    # Transform each ID to details via async call
    details = csp.async_node(ids, fetch_details)
    csp.print("details", details)
```

**Key points:**

- Each input tick triggers an async transformation
- Output ticks when the async operation completes
- Order of outputs matches order of operation completion (may differ from input order)

## Node-Level Async Operations

These tools allow async operations inside CSP node definitions.

### await\_ - Blocking Await in Nodes

`csp.await_` allows blocking on an async operation inside a node. The node execution pauses until the async operation completes.

**Signature:**

```python
def await_(
    coro: Awaitable[T],
    block: bool = True,
    timeout: float = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> T
```

**Parameters:**

- `coro`: A coroutine instance (result of calling an async function)
- `block`: If True (default), blocks until the coroutine completes. If False, returns a Future
- `timeout`: Optional timeout in seconds
- `loop`: Event loop to use. If None, uses the best available loop (see [Event Loop Options](#event-loop-options))

**Example:**

```python
async def fetch_value(key: str) -> int:
    await asyncio.sleep(0.1)
    return hash(key) % 100

@csp.node
def node_with_await(key: ts[str]) -> ts[int]:
    if csp.ticked(key):
        # Block until async completes
        result = csp.await_(fetch_value(key), block=True)
        return result
```

**Key points:**

- `block=True` (default) blocks until completion
- `block=False` returns a `Future` for later checking
- Uses the appropriate event loop automatically

### async_alarm - Alarm-Like Async Pattern

`csp.async_alarm` provides the most idiomatic CSP pattern for async operations. It works like a regular alarm but fires when an async operation completes.

```python
async def async_process(value: int) -> int:
    await asyncio.sleep(0.1)
    return value * 2

@csp.node
def node_with_async_alarm() -> ts[int]:
    with csp.alarms():
        poll_alarm = csp.alarm(bool)
        async_alarm = csp.async_alarm(int)  # Declare in alarms block

    with csp.state():
        s_counter = 0
        s_pending = False

    with csp.start():
        csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

    if csp.ticked(poll_alarm):
        # Only schedule new async if previous completed
        if not s_pending:
            s_counter += 1
            csp.schedule_async_alarm(async_alarm, async_process(s_counter))
            s_pending = True
        csp.schedule_alarm(poll_alarm, timedelta(milliseconds=10), True)

    if csp.ticked(async_alarm):
        # Async operation completed
        s_pending = False
        return async_alarm  # Returns the result value
```

**Key points:**

- Declare with `csp.async_alarm(T)` in the `with csp.alarms():` block
- Schedule with `csp.schedule_async_alarm(alarm, coroutine)`
- Check completion with `if csp.ticked(async_alarm):`
- Access the result value directly with `async_alarm`
- Lifecycle (start/stop) is managed automatically

### AsyncContext - Persistent Async Event Loop (Advanced)

> **Note:** `AsyncContext` is an internal class and not part of the public API. For most use cases, prefer `csp.async_alarm` which handles lifecycle automatically. If you need `AsyncContext`, import it from the implementation module.

`AsyncContext` provides a persistent async event loop for a node, avoiding the overhead of creating new loops for each operation.

```python
from csp.impl.async_adapter import AsyncContext

@csp.node
def node_with_context(x: ts[int]) -> ts[int]:
    with csp.state():
        s_ctx = None

    with csp.start():
        s_ctx = AsyncContext()
        s_ctx.start()

    with csp.stop():
        if s_ctx:
            s_ctx.stop()

    if csp.ticked(x):
        # Use the persistent event loop
        result = s_ctx.run(fetch_value(str(x)))
        return result
```

**Key points:**

- Import from `csp.impl.async_adapter` (not part of public API)
- Reuses the same event loop across ticks (more efficient)
- Must call `start()` in `with csp.start()` and `stop()` in `with csp.stop()`
- `run()` blocks until completion, `run_nowait()` returns a Future

## Complete Example

Here's a complete example demonstrating multiple async features:

```python
import csp
from csp import ts
from datetime import timedelta
from typing import AsyncIterator
import asyncio


async def async_fetch() -> int:
    """Single async fetch."""
    await asyncio.sleep(0.1)
    return 42


async def async_double(n: int) -> int:
    """Async transformation."""
    await asyncio.sleep(0.1)
    return n * 2


async def async_log(n: int) -> None:
    """Async side effect."""
    await asyncio.sleep(0.05)
    print(f"Logged: {n}")


async def async_stream(count: int) -> AsyncIterator[int]:
    """Async generator stream."""
    for i in range(count):
        await asyncio.sleep(0.1)
        yield i


@csp.node
def counter_node() -> ts[int]:
    with csp.alarms():
        tick = csp.alarm(bool)

    with csp.state():
        s_count = 0

    with csp.start():
        csp.schedule_alarm(tick, timedelta(), True)

    if csp.ticked(tick):
        s_count += 1
        csp.schedule_alarm(tick, timedelta(seconds=0.1), True)
        return s_count


@csp.node
def async_alarm_node() -> ts[int]:
    """Node using async_alarm pattern."""
    with csp.alarms():
        poll = csp.alarm(bool)
        result_alarm = csp.async_alarm(int)

    with csp.state():
        s_counter = 0
        s_pending = False

    with csp.start():
        csp.schedule_alarm(poll, timedelta(milliseconds=10), True)

    if csp.ticked(poll):
        if not s_pending:
            s_counter += 1
            csp.schedule_async_alarm(result_alarm, async_double(s_counter))
            s_pending = True
        csp.schedule_alarm(poll, timedelta(milliseconds=10), True)

    if csp.ticked(result_alarm):
        s_pending = False
        return result_alarm


@csp.graph
def main_graph():
    counter = counter_node()

    # async_for: stream from async generator
    stream = csp.async_for(async_stream(10))
    csp.print("stream", stream)

    # async_in: single async value
    initial = csp.async_in(async_fetch())
    csp.print("initial", initial)

    # async_out: async side effects
    csp.async_out(counter, async_log)

    # async_node: transform via async
    doubled = csp.async_node(counter, async_double)
    csp.print("doubled", doubled)

    # async_alarm in a node
    alarm_results = async_alarm_node()
    csp.print("alarm_results", alarm_results)


if __name__ == "__main__":
    csp.run(main_graph, realtime=True, endtime=timedelta(seconds=2))
```

### Running with Same-Thread Asyncio Mode

In realtime mode, CSP automatically runs async operations on the same thread (same-thread asyncio mode is the default):

```python
if __name__ == "__main__":
    # In realtime mode, same-thread asyncio is the default
    csp.run(main_graph, realtime=True, endtime=timedelta(seconds=2))

    # To use background thread for async instead:
    # csp.run(main_graph, realtime=True, endtime=timedelta(seconds=2), asyncio_on_thread=True)
```

## Best Practices

1. **Use `async_alarm` for node-internal async**: It's the most idiomatic CSP pattern and handles lifecycle automatically.

1. **Same-thread asyncio is the default**: In realtime mode, async operations run on the same thread as CSP, eliminating cross-thread overhead. Use `asyncio_on_thread=True` if you need the legacy background thread behavior.

1. **Track pending operations**: Use a state variable like `s_pending` to avoid scheduling overlapping async operations when order matters.

1. **Handle errors**: Async operations can fail. Consider wrapping in try/except:

   ```python
   try:
       result = csp.await_(risky_async_call(), block=True)
   except Exception as e:
       # Handle error
       pass
   ```

1. **Mind the timing**: Async operations complete at unpredictable times. If order matters, wait for one to complete before starting another.

1. **Use graph-level adapters when possible**: `async_for`, `async_in`, `async_out`, and `async_node` are simpler than node-internal async.

1. **Realtime mode required**: Async adapters only work in realtime mode (`realtime=True`). In simulation mode, async operations are not supported.

1. **Consider timeouts**: For blocking operations, consider using the `timeout` parameter:

   ```python
   result = csp.await_(slow_operation(), block=True, timeout=5.0)
   ```

1. **Event loop selection**: Let CSP automatically choose the best loop (default behavior). Only pass a custom `loop` parameter when you need to coordinate with external async code.

1. **Check asyncio mode when needed**: Use `is_csp_asyncio_mode()` to check if you're running inside CSP's asyncio mode, useful for conditional behavior in adapters or nodes.
