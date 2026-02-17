#!/usr/bin/env python
"""
Example usage of the Counter Adapter

This script demonstrates how to use the CounterAdapterManager to generate
and process sequential counter values using CSP's reactive programming model.

To run this example, first build the adapter:
    cd examples/05_cpp/3_cpp_adapter
    mkdir build && cd build
    cmake .. -DPYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    make -j

Then add the build directory to your Python path and run:
    PYTHONPATH=build:$PYTHONPATH python example.py
"""

import sys
from datetime import datetime, timedelta

import csp
from csp import ts


# For development/testing, add the build directory to path
# In production, you would install the module properly
def setup_path():
    """Add the build directory to the Python path if needed."""
    import os

    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.exists(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)


setup_path()

# Import the adapter after setting up the path
try:
    from counteradapter import CounterAdapterManager
except ImportError:
    print("Error: Could not import counteradapter.")
    print("Make sure you have built the C++ extension module.")
    print("See the build instructions in README.md")
    sys.exit(1)


@csp.node
def process_counter(value: ts[int]) -> ts[str]:
    """Process counter values and return a formatted string."""
    if csp.ticked(value):
        return f"Counter is at: {value}"


@csp.graph
def counter_graph() -> csp.OutputBasket(dict[str, csp.ts[int]]):
    """
    Example graph using the CounterAdapterManager.

    Creates a counter that ticks every 100ms and processes the values.
    """
    # Create the adapter manager with a 100ms interval
    counter_mgr = CounterAdapterManager(interval_ms=100, max_count=10)

    # Subscribe to counter values
    counter_values = counter_mgr.subscribe()

    # Process the values
    formatted = process_counter(counter_values)

    # Print the formatted values
    csp.print("Formatted", formatted)

    # Also publish to the output adapter (logs to stdout)
    counter_mgr.publish(counter_values)

    # Return the raw values for inspection
    return {"counter": counter_values}


def main():
    """Run the counter example."""
    print("Starting Counter Adapter Example")
    print("=" * 40)

    # Run the graph for 2 seconds
    start = datetime.utcnow()
    end = start + timedelta(seconds=2)

    result = csp.run(counter_graph, starttime=start, endtime=end, realtime=True)

    print("=" * 40)
    print(f"Received {len(result.get('counter', []))} counter values")

    if result.get("counter"):
        print(f"First value: {result['counter'][0]}")
        print(f"Last value: {result['counter'][-1]}")


if __name__ == "__main__":
    main()
