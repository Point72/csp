"""Pass CSP-parsed timeseries input metadata to numba_cfunc_compiler."""

import inspect
from dataclasses import dataclass
from typing import Any, Optional

from numba_cfunc_compiler.function_analyzer import InputTypeHandler
from numba_cfunc_compiler.models import ParameterInfo


@dataclass(frozen=True)
class CspInputMetadata:
    category: str


class CspInputHandler(InputTypeHandler):
    def try_parse(self, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        if not isinstance(ann, CspInputMetadata):
            return None
        return ParameterInfo(expected_type=ann, category=ann.category)

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        return value


def register():
    from numba_cfunc_compiler.function_analyzer import FunctionAnalyzer

    FunctionAnalyzer.register_input_handler(CspInputHandler())
