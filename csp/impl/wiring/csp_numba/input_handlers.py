"""Pass CSP-parsed timeseries input metadata to numba_cfunc_compiler."""

import inspect
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from numba_cfunc_compiler.core.analysis import InputTypeHandler, ParameterInfo


class CspInputCategory(Enum):
    SIGNAL = auto()
    SIGNAL_SET = auto()


@dataclass(frozen=True)
class CspInputMetadata:
    category: CspInputCategory


class CspInputHandler(InputTypeHandler):
    def try_parse(self, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        if not isinstance(ann, CspInputMetadata):
            return None
        return ParameterInfo(expected_type=ann, category=ann.category)

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        return value


def register():
    from numba_cfunc_compiler.core.analysis import FunctionAnalyzer

    FunctionAnalyzer.register_input_handler(CspInputHandler())
