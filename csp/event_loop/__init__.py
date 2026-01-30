"""
CSP Event Loop Integration

This module provides integration between CSP's event processing system and Python's asyncio.

There are two main integration patterns:

1. **Standalone Event Loop** (like uvloop):
   Use CSP as the asyncio event loop backend.

   ```python
   import csp.event_loop as csp_event_loop

   # Run using CSP's event loop
   csp_event_loop.run(my_coroutine())

   # Or use as a policy
   asyncio.set_event_loop_policy(csp_event_loop.EventLoopPolicy())
   ```

2. **Bridge with Running CSP Graph**:
   Interleave asyncio operations with a running CSP graph.

   ```python
   from csp.event_loop import AsyncioBridge

   bridge = AsyncioBridge(int, "my_data")

   @csp.graph
   def my_graph():
       data = bridge.adapter.out()
       csp.print("data", data)

   bridge.start()
   runner = csp.run_on_thread(my_graph, realtime=True, ...)

   # Schedule callbacks that push to CSP
   bridge.call_later(1.0, lambda: bridge.push(42))

   runner.join()
   bridge.stop()
   ```
"""

from csp.event_loop.bridge import AsyncioBridge, BidirectionalBridge
from csp.event_loop.loop import CspEventLoop, CspEventLoopPolicy, new_event_loop, run

__all__ = [
    # Standalone event loop
    "CspEventLoop",
    "CspEventLoopPolicy",
    "EventLoopPolicy",  # Alias for compatibility
    "new_event_loop",
    "run",
    # Bridge with running graph
    "AsyncioBridge",
    "BidirectionalBridge",
]

# Alias for compatibility with uvloop naming
EventLoopPolicy = CspEventLoopPolicy
