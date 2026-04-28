"""
CSP output handlers for numba_node.

Provides:
- SingleTsOutputHandler for ts[type] return annotation
- CspOutputsHandler for csp.Outputs(name=ts[type], ...) return annotation
"""

import ast
from typing import Any, Optional, get_args, get_origin

from csp.impl.types.tstype import TsType

from numba_type_utils.function_analyzer import OutputTypeHandler
from numba_type_utils.models import OutputAnalysis
from numba_type_utils.defaults import is_supported_type


def validate_output_type(t: Any, context: str) -> None:
    if not is_supported_type(t):
        raise TypeError(f"Unsupported type '{t}' for {context}")


def _extract_ts_inner_type(ann: Any) -> Optional[type]:
    if get_origin(ann) is not TsType:
        return None
    args = get_args(ann)
    return args[0] if args else getattr(ann, "typ", None)


class SingleTsOutputHandler(OutputTypeHandler):
    """Handles single ts[type] output annotations."""

    def try_parse(self, return_annotation: Any, ast_tree: ast.AST) -> Optional[OutputAnalysis]:
        if get_origin(return_annotation) is not TsType:
            return None

        output_type = _extract_ts_inner_type(return_annotation)
        if output_type is None:
            raise TypeError("Return type ts[type] is missing type argument.")

        validate_output_type(output_type, "output[0]")

        return OutputAnalysis(output_types=[output_type], named_outputs=None)


class CspOutputsHandler(OutputTypeHandler):
    """Handles csp.Outputs(name=ts[type], ...) for multiple named outputs."""

    def try_parse(self, return_annotation: Any, ast_tree: ast.AST) -> Optional[OutputAnalysis]:
        if not isinstance(return_annotation, type):
            return None
        if not hasattr(return_annotation, "__annotations__"):
            return None

        annotations = return_annotation.__annotations__
        if not annotations:
            return None

        named_outputs = {}

        for name, spec in annotations.items():
            if name == "__annotations__":
                continue

            output_type = _extract_ts_inner_type(spec)
            if output_type is None and hasattr(spec, "typ"):
                output_type = spec.typ
            if output_type is None:
                return None

            validate_output_type(output_type, f"output[{name}]")
            named_outputs[name] = output_type

        if not named_outputs:
            return None

        return OutputAnalysis(output_types=list(named_outputs.values()), named_outputs=named_outputs)


def register():
    from numba_type_utils.function_analyzer import FunctionAnalyzer

    FunctionAnalyzer.register_output_handler(SingleTsOutputHandler())
    FunctionAnalyzer.register_output_handler(CspOutputsHandler())
