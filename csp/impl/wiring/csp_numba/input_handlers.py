"""
CSP timeseries input handlers for numba_node.

Provides:
- TsInputHandler for ts[type] parameters
- ListBasketInputHandler for [ts[type]] list basket parameters
- DictBasketInputHandler for {key: ts[type]} dict basket parameters
"""

import inspect
from typing import Any, Optional, get_args, get_origin

from csp.impl.wiring.edge import Edge
from csp.impl.types.tstype import TsType

from numba_type_utils.function_analyzer import InputTypeHandler
from numba_type_utils.models import ParameterInfo


def _extract_ts_inner_type(ann: Any) -> Optional[type]:
    if get_origin(ann) is not TsType:
        return None
    args = get_args(ann)
    return args[0] if args else getattr(ann, "typ", None)


def _parse_list_basket_annotation(ann: Any) -> tuple[bool, Optional[type]]:
    # Legacy syntax: [ts[T]]
    if isinstance(ann, list):
        if len(ann) != 1:
            return True, None
        return True, _extract_ts_inner_type(ann[0])

    # Modern syntax: list[ts[T]] / typing.List[ts[T]]
    if get_origin(ann) is list:
        args = get_args(ann)
        if len(args) != 1:
            return True, None
        return True, _extract_ts_inner_type(args[0])

    return False, None


def _parse_dict_basket_annotation(ann: Any) -> tuple[bool, Optional[type]]:
    key_type = None
    value_ann = None

    # Legacy syntax: {str: ts[T]} / {int: ts[T]}
    if isinstance(ann, dict):
        if len(ann) != 1:
            return True, None
        key_type, value_ann = next(iter(ann.items()))
    # Modern syntax: dict[str, ts[T]] / typing.Dict[str, ts[T]]
    elif get_origin(ann) is dict:
        args = get_args(ann)
        if len(args) != 2:
            return True, None
        key_type, value_ann = args
    else:
        return False, None

    if key_type not in (str, int):
        return False, None

    return True, _extract_ts_inner_type(value_ann)


class TsInputHandler(InputTypeHandler):
    """Handles ts[type] input annotations."""

    def try_parse(self, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        inner_type = _extract_ts_inner_type(ann)
        if inner_type is None:
            return None

        return ParameterInfo(expected_type=inner_type, category="signal")

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        if not isinstance(value, Edge):
            raise TypeError(f"Argument '{param_name}' must be an Edge, got {type(value).__name__}")
        return value


class ListBasketInputHandler(InputTypeHandler):
    """Handles [ts[type]] and list[ts[type]] basket input annotations."""

    def try_parse(self, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        matched, inner_type = _parse_list_basket_annotation(ann)
        if not matched:
            return None
        if inner_type is None:
            raise TypeError(f"List basket '{param.name}' element ts[type] is missing type argument")

        return ParameterInfo(expected_type=inner_type, category="signal_set")

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"Argument '{param_name}' must be a list, got {type(value).__name__}")

        result = {}
        for i, edge in enumerate(value):
            if not isinstance(edge, Edge):
                raise TypeError(f"List basket '{param_name}[{i}]' must be an Edge, got {type(edge).__name__}")
            result[i] = edge
        return result


class DictBasketInputHandler(InputTypeHandler):
    """Handles {key_type: ts[type]} and dict[key_type, ts[type]] basket annotations."""

    def try_parse(self, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        matched, inner_type = _parse_dict_basket_annotation(ann)
        if not matched:
            return None
        if inner_type is None:
            raise TypeError(f"Dict basket '{param.name}' element ts[type] is missing type argument")

        return ParameterInfo(expected_type=inner_type, category="signal_set")

    def validate_value(self, param_name: str, value: Any, expected_type: Any) -> Any:
        if not isinstance(value, dict):
            raise TypeError(f"Argument '{param_name}' must be a dict, got {type(value).__name__}")

        result = {}
        for key, edge in value.items():
            if not isinstance(edge, Edge):
                raise TypeError(f"Dict basket '{param_name}[{key!r}]' must be an Edge, got {type(edge).__name__}")
            result[key] = edge
        return result


def register():
    from numba_type_utils.function_analyzer import FunctionAnalyzer

    FunctionAnalyzer.register_input_handler(TsInputHandler())
    FunctionAnalyzer.register_input_handler(ListBasketInputHandler())
    FunctionAnalyzer.register_input_handler(DictBasketInputHandler())
