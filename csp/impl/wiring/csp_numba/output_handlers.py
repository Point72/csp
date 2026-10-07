"""
CSP output handlers for numba_node.

Provides:
- SingleTsOutputHandler for ts[type] return annotation
- CspOutputsHandler for csp.Outputs(name=ts[type], ...) return annotation
"""

import ast
from typing import Any, Optional, get_args, get_origin

from numba_cfunc_compiler.core.analysis import OutputAnalysis, OutputTypeHandler
from numba_cfunc_compiler.types.factory import HostTypeFactory

from csp.impl.struct import Struct as CspStruct
from csp.impl.types.tstype import TsType


def validate_output_type(t: Any, context: str) -> None:
    if isinstance(t, type) and issubclass(t, CspStruct) and t is not CspStruct:
        raise TypeError(f"Unsupported type '{t}' for {context}")
    try:
        HostTypeFactory.resolve(t)
    except TypeError as exc:
        raise TypeError(f"Unsupported type '{t}' for {context}") from exc


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

        return OutputAnalysis(outputs={None: output_type})


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

        return OutputAnalysis(outputs=named_outputs)


def register():
    from numba_cfunc_compiler.core.analysis import FunctionAnalyzer

    FunctionAnalyzer.register_output_handler(SingleTsOutputHandler())
    FunctionAnalyzer.register_output_handler(CspOutputsHandler())
