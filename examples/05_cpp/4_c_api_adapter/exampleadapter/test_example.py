"""Tests for the C API adapter example."""

from datetime import datetime, timedelta

import csp

from . import _exampleadapterimpl
from .__main__ import main, managed_graph, standalone_graph


def test_standalone_adapters():
    """Test standalone input/output adapters."""
    csp.run(standalone_graph, starttime=datetime.now(), endtime=timedelta(milliseconds=200))


def test_managed_adapters():
    """Test managed adapters with adapter manager."""
    csp.run(managed_graph, starttime=datetime.now(), endtime=timedelta(milliseconds=200))


def test_struct_inspection():
    """Test struct inspection via C API."""

    class TestStruct(csp.Struct):
        name: str
        value: int
        price: float

    result = _exampleadapterimpl._example_inspect_struct_type(TestStruct)

    assert result["name"] == "TestStruct"
    assert result["field_count"] == 3
    assert "is_strict" in result
    assert len(result["fields"]) == 3

    field_names = [f["name"] for f in result["fields"]]
    assert "name" in field_names
    assert "value" in field_names
    assert "price" in field_names


def test_main():
    """Test the main entry point."""
    main()
