"""
CSP Event Loop Integration Examples

This module demonstrates how to use CSP's event loop integration to run
asynciocoroutines with CSP's event loop, similar to uvloop.
"""

import asyncio
import time

import csp.event_loop as csp_event_loop

# Example 1: Basic Usage


async def hello_world():
    """Simple async function."""
    print("Hello from CSP asyncio!")
    await asyncio.sleep(0.1)
    return "done"


def example_basic():
    """Basic usage of csp.event_loop.run()."""
    print("=" * 50)
    print("Example 1: Basic Usage")
    print("=" * 50)

    result = csp_event_loop.run(hello_world())
    print(f"Result: {result}\n")


# Example 2: Callback Scheduling


def example_callbacks():
    """Demonstrate callback scheduling methods."""
    print("=" * 50)
    print("Example 2: Callback Scheduling")
    print("=" * 50)

    results = []

    async def callback_demo():
        loop = asyncio.get_running_loop()

        def callback(value):
            results.append(value)
            print(f"Callback called with: {value}")

        # Schedule callbacks
        loop.call_soon(callback, "immediate")
        loop.call_later(0.1, callback, "delayed_100ms")
        loop.call_later(0.05, callback, "delayed_50ms")

        # Wait for all callbacks to complete
        await asyncio.sleep(0.2)

    csp_event_loop.run(callback_demo())

    print(f"Callback order: {results}\n")


# Example 3: Concurrent Tasks


async def fetch_data(name, delay):
    """Simulate fetching data with a delay."""
    print(f"Fetching {name}...")
    await asyncio.sleep(delay)
    print(f"Got {name}!")
    return f"data from {name}"


async def concurrent_tasks():
    """Run multiple tasks concurrently."""
    # Create tasks
    tasks = [
        asyncio.create_task(fetch_data("source1", 0.1)),
        asyncio.create_task(fetch_data("source2", 0.15)),
        asyncio.create_task(fetch_data("source3", 0.05)),
    ]

    # Wait for all tasks
    results = await asyncio.gather(*tasks)
    return results


def example_concurrent():
    """Demonstrate concurrent task execution."""
    print("=" * 50)
    print("Example 3: Concurrent Tasks")
    print("=" * 50)

    results = csp_event_loop.run(concurrent_tasks())
    print(f"Results: {results}\n")


# Example 4: Synchronization Primitives


async def worker_with_lock(lock, name, shared_data):
    """Worker that uses a lock to access shared data."""
    async with lock:
        print(f"{name}: Acquired lock")
        shared_data.append(f"{name} start")
        await asyncio.sleep(0.05)
        shared_data.append(f"{name} end")
        print(f"{name}: Releasing lock")


async def synchronization_example():
    """Demonstrate asyncio synchronization primitives."""
    lock = asyncio.Lock()
    shared_data = []

    await asyncio.gather(
        worker_with_lock(lock, "Worker1", shared_data),
        worker_with_lock(lock, "Worker2", shared_data),
        worker_with_lock(lock, "Worker3", shared_data),
    )

    return shared_data


def example_synchronization():
    """Demonstrate synchronization primitives."""
    print("=" * 50)
    print("Example 4: Synchronization Primitives")
    print("=" * 50)

    results = csp_event_loop.run(synchronization_example())
    print(f"Execution order: {results}\n")


# Example 5: Producer-Consumer


async def producer(queue, items):
    """Produce items and put them in a queue."""
    for item in items:
        await queue.put(item)
        print(f"Produced: {item}")
        await asyncio.sleep(0.02)
    await queue.put(None)  # Signal end


async def consumer(queue):
    """Consume items from a queue."""
    results = []
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        results.append(item)
    return results


async def producer_consumer():
    """Producer-consumer pattern."""
    queue = asyncio.Queue()

    # Run producer and consumer concurrently
    producer_task = asyncio.create_task(producer(queue, [1, 2, 3, 4, 5]))
    consumer_task = asyncio.create_task(consumer(queue))

    await producer_task
    results = await consumer_task

    return results


def example_producer_consumer():
    """Demonstrate producer-consumer pattern."""
    print("=" * 50)
    print("Example 5: Producer-Consumer Pattern")
    print("=" * 50)

    results = csp_event_loop.run(producer_consumer())
    print(f"All consumed items: {results}\n")


# Example 6: Timeout Handling


async def slow_operation():
    """A slow operation that might need to be cancelled."""
    await asyncio.sleep(10)
    return "completed"


async def timeout_example():
    """Demonstrate timeout handling."""
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=0.1)
        return result
    except asyncio.TimeoutError:
        return "operation timed out"


def example_timeout():
    """Demonstrate timeout handling."""
    print("=" * 50)
    print("Example 6: Timeout Handling")
    print("=" * 50)

    result = csp_event_loop.run(timeout_example())
    print(f"Result: {result}\n")


# Example 7: Event Loop Policy


def example_policy():
    """Demonstrate using CSP as the asyncio event loop policy."""
    print("=" * 50)
    print("Example 7: Event Loop Policy")
    print("=" * 50)

    # Save old policy
    old_policy = asyncio.get_event_loop_policy()

    try:
        # Set CSP as the event loop policy
        asyncio.set_event_loop_policy(csp_event_loop.EventLoopPolicy())

        # Now asyncio.run() uses CSP's event loop
        async def check_loop():
            loop = asyncio.get_running_loop()
            return type(loop).__name__

        loop_name = asyncio.run(check_loop())
        print(f"Using event loop: {loop_name}\n")
    finally:
        # Restore old policy
        asyncio.set_event_loop_policy(old_policy)


# Example 8: Exception Handling


async def faulty_operation():
    """An operation that raises an exception."""
    await asyncio.sleep(0.01)
    raise ValueError("Something went wrong!")


async def exception_example():
    """Demonstrate exception handling in async code."""
    try:
        await faulty_operation()
    except ValueError as e:
        return f"Caught exception: {e}"


def example_exception_handling():
    """Demonstrate exception handling."""
    print("=" * 50)
    print("Example 8: Exception Handling")
    print("=" * 50)

    result = csp_event_loop.run(exception_example())
    print(f"Result: {result}\n")


# Example 9: Executor


def blocking_io_operation(x, y):
    """A blocking I/O operation."""
    time.sleep(0.1)  # Simulate blocking I/O
    return x * y


async def executor_example():
    """Run blocking code in an executor."""
    loop = asyncio.get_running_loop()

    # Run blocking operations in the default executor with timeout
    try:
        result1 = await asyncio.wait_for(loop.run_in_executor(None, blocking_io_operation, 5, 10), timeout=5.0)
        result2 = await asyncio.wait_for(loop.run_in_executor(None, blocking_io_operation, 3, 7), timeout=5.0)
    except asyncio.TimeoutError:
        return "executor timed out", None

    return result1, result2


def example_executor():
    """Demonstrate running blocking code in executor."""
    print("=" * 50)
    print("Example 9: Running Blocking Code in Executor")
    print("=" * 50)

    results = csp_event_loop.run(executor_example())
    print(f"Results: {results}\n")


def _check_event_loop_available():
    """Check if CSP event loop is functional by checking required methods exist."""
    try:
        loop = csp_event_loop.new_event_loop()
        # Check that the CSP engine has the required start method
        if not hasattr(loop, "_csp_engine"):
            loop.close()
            return False
        if not hasattr(loop._csp_engine, "start"):
            loop.close()
            return False
        loop.close()
        return True
    except Exception:
        return False


def main():
    """Run all examples."""
    print("\n" + "=" * 50)
    print("CSP ASYNCIO INTEGRATION EXAMPLES")
    print("=" * 50 + "\n")

    # Check if event loop is functional
    if not _check_event_loop_available():
        print("CSP event loop is not available or functional.")
        print("Skipping asyncio integration examples.")
        print("=" * 50)
        return

    examples = [
        ("basic", example_basic),
        ("callbacks", example_callbacks),
        ("concurrent", example_concurrent),
        ("synchronization", example_synchronization),
        ("producer_consumer", example_producer_consumer),
        ("timeout", example_timeout),
        ("policy", example_policy),
        ("exception_handling", example_exception_handling),
        ("executor", example_executor),
    ]

    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"Example {name} failed: {e}\n")

    print("=" * 50)
    print("All examples completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
