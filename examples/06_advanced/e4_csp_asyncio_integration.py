"""
CSP Event Loop Integration with Running Graph

This example demonstrates how to integrate asyncio event loop operations
with a running CSP graph using the AsyncioBridge and BidirectionalBridge
classes from csp.event_loop.

Examples show how to:
1. Run asyncio coroutines alongside a CSP graph in realtime mode
2. Use push adapters to feed data from asyncio callbacks into CSP
3. Interleave operations with the CSP engine and its clock (csp.now())
4. Use call_later and call_at to schedule callbacks that interact with CSP
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Callable, List, Optional

import csp
from csp import ts
from csp.event_loop import AsyncioBridge, BidirectionalBridge
from csp.utils.datetime import utc_now

# Example 1: Basic Push from Asyncio to CSP


def example_basic_push():
    """Demonstrate pushing data from asyncio callbacks to CSP graph."""
    print("=" * 60)
    print("Example 1: Basic Push from Asyncio to CSP")
    print("=" * 60)

    bridge = AsyncioBridge(int, "counter")

    @csp.node
    def collect(data: ts[int]) -> ts[int]:
        if csp.ticked(data):
            print(f"  CSP received at {csp.now()}: {data}")
            return data

    @csp.graph
    def g():
        data = bridge.adapter.out()
        collected = collect(data)
        csp.add_graph_output("data", collected)

    # Start bridge first
    start_time = utc_now()
    bridge.start(start_time)

    # Give bridge time to start
    time.sleep(0.1)

    # Run CSP graph in thread
    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=2))

    # Wait for adapter to bind
    bridge.adapter.wait_for_start(timeout=1.0)

    # Push data from asyncio
    counter = 0

    def push_counter():
        nonlocal counter
        if bridge._running and runner.is_alive():
            bridge.push(counter)
            counter += 1
            if counter < 5:
                bridge.call_later(0.3, push_counter)

    bridge.call_soon(push_counter)

    # Wait for completion
    results = runner.join()
    bridge.stop()

    print(f"  Collected: {[v for _, v in results.get('data', [])]}\n")


# Example 2: Scheduled Callbacks with call_later


def example_call_later():
    """Demonstrate call_later scheduling with CSP."""
    print("=" * 60)
    print("Example 2: Scheduled Callbacks with call_later")
    print("=" * 60)

    bridge = AsyncioBridge(str, "messages")

    @csp.node
    def log_events(msg: ts[str]) -> ts[str]:
        if csp.ticked(msg):
            now = csp.now()
            print(f"  [{now}] CSP: {msg}")
            return msg

    @csp.graph
    def g():
        data = bridge.adapter.out()
        logged = log_events(data)
        csp.add_graph_output("events", logged)

    start_time = utc_now()
    bridge.start(start_time)
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=3))

    bridge.adapter.wait_for_start(timeout=1.0)

    # Schedule messages at different times
    bridge.call_later(0.1, lambda: bridge.push("Message at 0.1s"))
    bridge.call_later(0.5, lambda: bridge.push("Message at 0.5s"))
    bridge.call_later(1.0, lambda: bridge.push("Message at 1.0s"))
    bridge.call_later(1.5, lambda: bridge.push("Message at 1.5s"))
    bridge.call_later(2.0, lambda: bridge.push("Message at 2.0s"))

    results = runner.join()
    bridge.stop()

    print(f"  Total messages: {len(results.get('events', []))}\n")


# Example 3: Call at Specific Times


def example_call_at():
    """Demonstrate call_at scheduling at specific datetimes."""
    print("=" * 60)
    print("Example 3: Call at Specific Datetimes")
    print("=" * 60)

    bridge = AsyncioBridge(str, "timed_events")

    @csp.node
    def process(msg: ts[str]) -> ts[str]:
        if csp.ticked(msg):
            print(f"  [{csp.now().strftime('%H:%M:%S.%f')}] {msg}")
            return msg

    @csp.graph
    def g():
        data = bridge.adapter.out()
        processed = process(data)
        csp.add_graph_output("events", processed)

    start_time = utc_now()
    bridge.start(start_time)
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=3))

    bridge.adapter.wait_for_start(timeout=1.0)

    # Schedule at specific times
    t1 = start_time + timedelta(seconds=0.5)
    t2 = start_time + timedelta(seconds=1.0)
    t3 = start_time + timedelta(seconds=1.5)

    bridge.call_at(t1, lambda: bridge.push(f"Event scheduled for {t1.strftime('%H:%M:%S.%f')}"))
    bridge.call_at(t2, lambda: bridge.push(f"Event scheduled for {t2.strftime('%H:%M:%S.%f')}"))
    bridge.call_at(t3, lambda: bridge.push(f"Event scheduled for {t3.strftime('%H:%M:%S.%f')}"))

    results = runner.join()
    bridge.stop()

    print(f"  Total events: {len(results.get('events', []))}\n")


# Example 4: Running Async Coroutines


def example_async_coroutines():
    """Demonstrate running asyncio coroutines that push to CSP."""
    print("=" * 60)
    print("Example 4: Running Async Coroutines")
    print("=" * 60)

    bridge = AsyncioBridge(dict, "async_data")

    @csp.node
    def process_data(data: ts[dict]) -> ts[str]:
        if csp.ticked(data):
            result = f"Processed: {data}"
            print(f"  CSP [{csp.now()}]: {result}")
            return result

    @csp.graph
    def g():
        data = bridge.adapter.out()
        processed = process_data(data)
        csp.add_graph_output("results", processed)

    start_time = utc_now()
    bridge.start(start_time)
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=3))

    bridge.adapter.wait_for_start(timeout=1.0)

    # Define an async coroutine that fetches data
    async def fetch_data(source: str, delay: float) -> dict:
        await asyncio.sleep(delay)
        return {"source": source, "value": delay * 100, "timestamp": time.time()}

    # Run multiple coroutines and push results to CSP
    async def fetch_and_push():
        for i in range(3):
            data = await fetch_data(f"source_{i}", 0.3 + i * 0.2)
            bridge.push(data)

    bridge.run_coroutine(fetch_and_push())

    results = runner.join()
    bridge.stop()

    print(f"  Total results: {len(results.get('results', []))}\n")


# Example 5: Bidirectional Communication


def example_bidirectional():
    """Demonstrate bidirectional communication between asyncio and CSP."""
    print("=" * 60)
    print("Example 5: Bidirectional Communication")
    print("=" * 60)

    bridge = BidirectionalBridge()
    received_in_async = []

    @csp.node
    def process_and_respond(data: ts[object], bridge_ref: object) -> ts[str]:
        if csp.ticked(data):
            print(f"  CSP: Processing {data}")
            # Emit response back to asyncio
            bridge_ref.emit({"original": data, "processed_at": str(csp.now())})
            return f"CSP processed: {data}"

    @csp.graph
    def g():
        data = bridge.adapter.out()
        result = process_and_respond(data, bridge)
        csp.add_graph_output("results", result)

    # Register callback to receive events from CSP
    def on_event(event):
        print(f"  Asyncio received: {event}")
        received_in_async.append(event)

    bridge.on_event(on_event)

    start_time = utc_now()
    bridge.start()
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=2))

    bridge.adapter.wait_for_start(timeout=1.0)

    # Push data to CSP
    for i in range(3):
        time.sleep(0.3)
        bridge.push(f"message_{i}")

    results = runner.join()
    bridge.stop()

    print(f"  CSP outputs: {len(results.get('results', []))}")
    print(f"  Asyncio received: {len(received_in_async)} events\n")


# Example 6: Periodic Tasks with CSP Timer Integration


def example_periodic_tasks():
    """Demonstrate periodic async tasks alongside CSP timers."""
    print("=" * 60)
    print("Example 6: Periodic Tasks with CSP Timer Integration")
    print("=" * 60)

    bridge = AsyncioBridge(str, "async_ticks")

    @csp.node
    def csp_timer_node(timer: ts[bool], async_data: ts[str]) -> ts[str]:
        """Node that processes both CSP timer ticks and async data."""
        with csp.state():
            s_count = 0

        if csp.ticked(timer):
            s_count += 1
            msg = f"CSP timer tick #{s_count} at {csp.now()}"
            print(f"  {msg}")
            return msg

        if csp.ticked(async_data):
            msg = f"Async data: {async_data} at {csp.now()}"
            print(f"  {msg}")
            return msg

    @csp.graph
    def g():
        # CSP's own timer
        timer = csp.timer(timedelta(milliseconds=400))
        # Async data coming in
        async_data = bridge.adapter.out()
        # Process both
        result = csp_timer_node(timer, async_data)
        csp.add_graph_output("events", result)

    start_time = utc_now()
    bridge.start(start_time)
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=2))

    bridge.adapter.wait_for_start(timeout=1.0)

    # Set up periodic async task
    tick_count = [0]

    def periodic_tick():
        if bridge._running and runner.is_alive():
            tick_count[0] += 1
            bridge.push(f"async_tick_{tick_count[0]}")
            if tick_count[0] < 5:
                bridge.call_later(0.3, periodic_tick)

    bridge.call_later(0.15, periodic_tick)  # Offset from CSP timer

    results = runner.join()
    bridge.stop()

    print(f"  Total events: {len(results.get('events', []))}\n")


# Example 7: Using csp.now() Time for Scheduling


class CspAwareScheduler:
    """
    A scheduler that can coordinate with CSP's internal time.

    This demonstrates how to use CSP's time (csp.now()) as a reference
    for scheduling asyncio callbacks.
    """

    def __init__(self):
        self.bridge = AsyncioBridge(dict, "scheduler_events")
        self._csp_start_time: Optional[datetime] = None
        self._pending_schedules: List[tuple] = []

    def set_csp_start_time(self, start_time: datetime) -> None:
        """Set the CSP engine start time for time calculations."""
        self._csp_start_time = start_time
        self.bridge.start(start_time)

    def schedule_at_csp_offset(self, offset: timedelta, callback: Callable[..., Any], *args: Any) -> None:
        """
        Schedule a callback at a specific offset from CSP start time.

        This ensures the callback fires at a time aligned with csp.now().
        """
        if self._csp_start_time is None:
            self._pending_schedules.append((offset, callback, args))
            return

        target_time = self._csp_start_time + offset
        self.bridge.call_at(target_time, callback, *args)

    def push_with_metadata(self, event_type: str, data: Any) -> None:
        """Push data with timing metadata."""
        self.bridge.push(
            {
                "type": event_type,
                "data": data,
                "wall_time": datetime.utcnow().isoformat(),
            }
        )


def example_csp_time_scheduling():
    """Demonstrate scheduling aligned with CSP time."""
    print("=" * 60)
    print("Example 7: Scheduling Aligned with CSP Time")
    print("=" * 60)

    scheduler = CspAwareScheduler()

    @csp.node
    def process(event: ts[dict]) -> ts[str]:
        if csp.ticked(event):
            csp_time = csp.now()
            wall_time = event.get("wall_time", "unknown")
            event_type = event.get("type", "unknown")
            data = event.get("data", None)
            result = f"[{csp_time}] {event_type}: {data} (wall: {wall_time[-12:]})"
            print(f"  {result}")
            return result

    @csp.graph
    def g():
        data = scheduler.bridge.adapter.out()
        processed = process(data)
        csp.add_graph_output("events", processed)

    start_time = utc_now()
    scheduler.set_csp_start_time(start_time)
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=3))

    scheduler.bridge.adapter.wait_for_start(timeout=1.0)

    # Schedule events at specific CSP time offsets
    scheduler.schedule_at_csp_offset(
        timedelta(milliseconds=200),
        lambda: scheduler.push_with_metadata("tick", "event_1"),
    )
    scheduler.schedule_at_csp_offset(
        timedelta(milliseconds=600),
        lambda: scheduler.push_with_metadata("tick", "event_2"),
    )
    scheduler.schedule_at_csp_offset(
        timedelta(seconds=1),
        lambda: scheduler.push_with_metadata("milestone", "1 second mark"),
    )
    scheduler.schedule_at_csp_offset(
        timedelta(seconds=1, milliseconds=500),
        lambda: scheduler.push_with_metadata("tick", "event_3"),
    )

    results = runner.join()
    scheduler.bridge.stop()

    print(f"  Total events: {len(results.get('events', []))}\n")


# Example 8: Error Handling


def example_error_handling():
    """Demonstrate error handling in asyncio-CSP integration."""
    print("=" * 60)
    print("Example 8: Error Handling")
    print("=" * 60)

    bridge = AsyncioBridge(str, "error_demo")
    errors_caught = []

    @csp.node
    def process(data: ts[str]) -> ts[str]:
        if csp.ticked(data):
            if "error" in data.lower():
                raise ValueError(f"Error processing: {data}")
            print(f"  CSP processed: {data}")
            return data

    @csp.graph
    def g():
        data = bridge.adapter.out()
        # Note: In production, you'd want proper error handling here
        processed = process(data)
        csp.add_graph_output("results", processed)

    start_time = utc_now()
    bridge.start(start_time)
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=2))

    bridge.adapter.wait_for_start(timeout=1.0)

    # Push normal data
    bridge.call_later(0.1, lambda: bridge.push("normal message 1"))
    bridge.call_later(0.3, lambda: bridge.push("normal message 2"))
    bridge.call_later(0.5, lambda: bridge.push("normal message 3"))

    # Demonstrate async error handling
    async def async_operation_with_error():
        try:
            await asyncio.sleep(0.2)
            raise RuntimeError("Simulated async error")
        except RuntimeError as e:
            print(f"  Caught async error: {e}")
            errors_caught.append(str(e))
            bridge.push("recovered from error")

    bridge.run_coroutine(async_operation_with_error())

    try:
        results = runner.join()
    except Exception as e:
        print(f"  Graph error (expected): {e}")
        results = {}

    bridge.stop()

    print(f"  Errors caught in async: {len(errors_caught)}")
    print(f"  Results: {len(results.get('results', []))}\n")


# Example 9: Integration with Standard Asyncio Libraries


def example_asyncio_libraries():
    """Demonstrate using standard asyncio patterns with CSP."""
    print("=" * 60)
    print("Example 9: Integration with Standard Asyncio Libraries")
    print("=" * 60)

    bridge = AsyncioBridge(dict, "asyncio_events")

    @csp.node
    def aggregate(events: ts[dict]) -> ts[dict]:
        """Aggregate events from asyncio."""
        with csp.state():
            s_count = 0

        if csp.ticked(events):
            s_count += 1
            events["count"] = s_count
            events["csp_time"] = str(csp.now())
            print(f"  CSP: Event #{s_count} - {events.get('type')}")
            return events

    @csp.graph
    def g():
        data = bridge.adapter.out()
        agg = aggregate(data)
        csp.add_graph_output("aggregated", agg)

    start_time = utc_now()
    bridge.start(start_time)
    time.sleep(0.1)

    runner = csp.run_on_thread(g, realtime=True, starttime=start_time, endtime=timedelta(seconds=3))

    bridge.adapter.wait_for_start(timeout=1.0)

    # Use standard asyncio patterns
    async def producer_consumer_pattern():
        """Demonstrate asyncio Queue with CSP."""
        queue = asyncio.Queue()

        async def producer():
            for i in range(5):
                await asyncio.sleep(0.2)
                await queue.put({"type": "produced", "value": i})

        async def consumer():
            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    bridge.push(item)
                except asyncio.TimeoutError:
                    break

        await asyncio.gather(producer(), consumer())

    async def gather_pattern():
        """Demonstrate asyncio.gather with CSP."""

        async def task(name, delay):
            await asyncio.sleep(delay)
            return {"type": "completed", "task": name, "delay": delay}

        results = await asyncio.gather(
            task("fast", 0.1),
            task("medium", 0.2),
            task("slow", 0.3),
        )
        for result in results:
            bridge.push(result)

    # Run both patterns
    bridge.run_coroutine(producer_consumer_pattern())
    bridge.run_coroutine(gather_pattern())

    results = runner.join()
    bridge.stop()

    print(f"  Total aggregated events: {len(results.get('aggregated', []))}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("CSP ASYNCIO INTEGRATION WITH RUNNING GRAPH")
    print("=" * 60 + "\n")

    example_basic_push()
    example_call_later()
    example_call_at()
    example_async_coroutines()
    example_bidirectional()
    example_periodic_tasks()
    example_csp_time_scheduling()
    example_error_handling()
    example_asyncio_libraries()

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
