"""
Example demonstrating the CSP C API adapters.

This example shows how to use adapters written in C via the C API:
- ExampleOutputAdapter: Prints received values to stdout
- ExamplePushInputAdapter: Generates incrementing integers in a background thread

These adapters are compiled separately from CSP and communicate via the
ABI-stable C interface defined in cpp/csp/engine/c/*.h headers.

See docs/wiki/how-tos/Write-C-API-Adapters.md for how to write your own.
"""

from datetime import timedelta

import csp
from csp import ts
from csp.adapters.c_example import _example_input_adapter_def, _example_output_adapter_def
from csp.utils.datetime import utc_now


# Example 1: Using the C output adapter to print values
@csp.graph
def output_adapter_example():
    """
    Demonstrates the C output adapter which receives values and prints them.
    The adapter is implemented in C in cpp/csp/adapters/c/example/ExampleOutputAdapter.c
    """
    # Create a simple curve of values
    data = csp.curve(
        typ=int,
        data=[
            (timedelta(milliseconds=100), 1),
            (timedelta(milliseconds=200), 2),
            (timedelta(milliseconds=300), 3),
            (timedelta(milliseconds=400), 4),
            (timedelta(milliseconds=500), 5),
        ],
    )

    # Also print using Python's csp.print for comparison
    csp.print("Python print", data)

    # Use the C output adapter - it will print values with a prefix
    _example_output_adapter_def(data)


# Example 2: Using the C push input adapter to receive values from C
@csp.graph
def input_adapter_example() -> ts[int]:
    """
    Demonstrates the C push input adapter which generates values in a background thread.
    The adapter is implemented in C in cpp/csp/adapters/c/example/ExamplePushInputAdapter.c
    """
    # Create input adapter that generates integers every 100ms
    data = _example_input_adapter_def(typ=int, properties={"interval_ms": 100})

    # Print the received values
    csp.print("From C adapter", data)

    return data


# Example 3: Both adapters together - C input -> CSP -> C output
@csp.graph
def full_pipeline_example():
    """
    Demonstrates a full pipeline: C adapter produces data, CSP processes it,
    and another C adapter consumes it.
    """
    # Receive integers from C background thread
    raw_data = _example_input_adapter_def(typ=int, properties={"interval_ms": 50})

    # Process in CSP - double the values
    processed = raw_data * 2

    # Print using Python for debugging
    csp.print("raw", raw_data)
    csp.print("processed", processed)

    # Send to C output adapter
    _example_output_adapter_def(processed)


def run_output_adapter_example():
    """Run the output adapter example."""
    print("\n" + "=" * 60)
    print("Example 1: C Output Adapter")
    print("=" * 60)
    print("Running graph with C output adapter...")
    print()

    csp.run(
        output_adapter_example,
        starttime=utc_now(),
        endtime=timedelta(seconds=1),
        realtime=False,  # Sim mode
    )


def run_input_adapter_example():
    """Run the input adapter example."""
    print("\n" + "=" * 60)
    print("Example 2: C Push Input Adapter")
    print("=" * 60)
    print("Running graph with C input adapter (realtime, 500ms)...")
    print()

    csp.run(
        input_adapter_example,
        starttime=utc_now(),
        endtime=timedelta(milliseconds=500),
        realtime=True,  # Need realtime for push adapters
    )


def run_full_pipeline_example():
    """Run the full pipeline example."""
    print("\n" + "=" * 60)
    print("Example 3: Full C -> CSP -> C Pipeline")
    print("=" * 60)
    print("Running full pipeline (realtime, 300ms)...")
    print()

    csp.run(
        full_pipeline_example,
        starttime=utc_now(),
        endtime=timedelta(milliseconds=300),
        realtime=True,
    )


def main():
    """Run all examples."""
    print("CSP C API Adapter Examples")
    print("=" * 60)

    # Only run the output adapter example by default since the input
    # adapter requires the C library to be compiled with threading support
    run_output_adapter_example()

    # Uncomment to run input adapter examples:
    # run_input_adapter_example()
    # run_full_pipeline_example()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
