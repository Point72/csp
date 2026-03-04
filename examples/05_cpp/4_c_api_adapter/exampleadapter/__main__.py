"""
Example C API adapter usage.

This module demonstrates using C adapters implemented via the CSP C API.
It shows two patterns:

1. Standalone adapters - Simple adapters without lifecycle coordination
2. Managed adapters - Adapters using an AdapterManager for coordinated lifecycles
"""

from datetime import datetime, timedelta

import csp
from exampleadapter import ExampleAdapterManager, example_input, example_output


@csp.graph
def standalone_graph():
    """Graph using standalone C API adapters (no manager)."""
    data = example_input(int, interval_ms=100)
    example_output(data, prefix="[Standalone] ")


@csp.graph
def managed_graph():
    """
    Graph using managed C API adapters (with adapter manager).

    The adapter manager coordinates the lifecycle of multiple adapters:
    - All adapters start/stop together
    - Push groups enable batched event processing
    - Shared configuration (like prefix) across adapters
    """
    # Create an adapter manager with a common prefix
    mgr = ExampleAdapterManager(prefix="[Managed] ")

    # Subscribe to data through the manager
    data = mgr.subscribe(int, interval_ms=100)

    # Publish through the manager
    mgr.publish(data)


def demo_standalone():
    """Demonstrate standalone adapters."""
    print("=== Standalone Adapters ===")
    print("Using adapters without an adapter manager.")
    print("Each adapter has independent lifecycle.\n")
    csp.run(standalone_graph, starttime=datetime.now(), endtime=timedelta(milliseconds=300))


def demo_managed():
    """Demonstrate managed adapters."""
    print("\n=== Managed Adapters ===")
    print("Using adapters with an ExampleAdapterManager.")
    print("All adapters share coordinated lifecycle and configuration.\n")
    csp.run(managed_graph, starttime=datetime.now(), endtime=timedelta(milliseconds=300))


def main():
    """Run the C API adapter examples."""
    demo_standalone()
    demo_managed()
    print("\n=== Done ===")


if __name__ == "__main__":
    main()
