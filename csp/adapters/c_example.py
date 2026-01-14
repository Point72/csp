from typing import TypeVar

from csp import ts
from csp.impl.wiring import input_adapter_def, output_adapter_def
from csp.lib import _exampleadapterimpl

T = TypeVar("T")

_example_input_adapter_def = input_adapter_def(
    "example_input_adapter",
    _exampleadapterimpl._example_input_adapter,
    ts["T"],
    typ="T",
    properties=dict,
)

_example_output_adapter_def = output_adapter_def(
    "example_output_adapter",
    _exampleadapterimpl._example_output_adapter,
    input=ts["T"],
)
