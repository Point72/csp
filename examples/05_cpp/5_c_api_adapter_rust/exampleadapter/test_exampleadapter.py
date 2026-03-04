"""Tests for the Rust adapter example."""

from datetime import datetime, timedelta

import csp

from .__main__ import main, managed_graph, standalone_graph


def test_standalone_adapters():
    """Test standalone Rust input/output adapters."""
    csp.run(standalone_graph, starttime=datetime.now(), endtime=timedelta(milliseconds=200))


def test_managed_adapters():
    """Test managed adapters with adapter manager."""
    csp.run(managed_graph, starttime=datetime.now(), endtime=timedelta(milliseconds=200))


def test_main():
    """Test the main entry point."""
    main()
