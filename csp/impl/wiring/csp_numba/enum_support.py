import ast
import inspect
from dataclasses import dataclass
from typing import Any, Optional

from csp.impl.enum import Enum as CspEnum

from numba_type_utils.type_factory import TypeFactory
from numba_type_utils.models import VariableType, ParameterInfo, StateVariableInfo
from numba_type_utils.numba_type_inference import NumbaTypeInference
from numba_type_utils.utils.ast import AST


def _resolve_csp_enum_member(node: ast.AST, globalns: dict) -> Optional[CspEnum]:
    """Resolve `MyEnum.VALUE` and `MyEnum.VALUE.value` AST nodes."""
    if not isinstance(node, ast.Attribute):
        return None

    if node.attr == "value":
        return _resolve_csp_enum_member(node.value, globalns)

    if not isinstance(node.value, (ast.Name, ast.Attribute)):
        return None

    try:
        enum_class = eval(ast.unparse(node.value), globalns)
    except Exception:
        return None

    if not (isinstance(enum_class, type) and issubclass(enum_class, CspEnum)):
        return None

    try:
        enum_member = getattr(enum_class, node.attr)
    except AttributeError:
        return None

    return enum_member if isinstance(enum_member, CspEnum) else None


@dataclass(frozen=True)
class CspEnumType(VariableType):
    """Handles csp.Enum types, represented as int64 in Numba."""

    def get_numba_type_name(self) -> str:
        # enums are represented as int64 in Numba
        return "int64"

    def get_methods(self):
        # enums have no methods
        return []

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return isinstance(var_type, type) and issubclass(var_type, CspEnum)

    @classmethod
    def from_type(cls, var_type: Any, value: Any) -> Optional["CspEnumType"]:
        """Create CspEnumType from a csp.Enum subclass."""
        if cls.is_type_supported(var_type):
            return cls(var_type, value)
        return None

    @classmethod
    def try_lower_assignment(
        cls, node: ast.Assign, rhs: ast.AST, call_globals: dict
    ) -> Optional[tuple[list, "CspEnumType"]]:
        """Lower: x = MyEnum.VALUE → x = <int64_value>"""
        enum_member = _resolve_csp_enum_member(rhs, call_globals)
        if enum_member is None:
            return None

        var_name = node.targets[0].id
        var_type = cls(type(enum_member), enum_member)

        return AST.assignment(var_name, ast.Constant(value=enum_member.value)), var_type

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann: Any) -> Optional[ParameterInfo]:
        """Parse csp.Enum constant input parameters."""
        if isinstance(ann, type) and issubclass(ann, CspEnum):
            return ParameterInfo(expected_type=ann)  # defaults to category="constant"
        return None

    @classmethod
    def validate_input(cls, param_name: str, value: Any, expected_type: Any) -> Any:
        """Validate and convert enum input to int64."""
        if not isinstance(value, expected_type):
            raise TypeError(f"Argument '{param_name}' expected {expected_type}, got {type(value)}")
        return value.value  # Convert to int64

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> Optional[StateVariableInfo]:
        """Parse State[MyEnum] = MyEnum.VALUE declarations."""
        slice_node = node.annotation.slice

        if isinstance(slice_node, ast.Name):
            state_type = globalns.get(slice_node.id)
            if state_type is None or not isinstance(state_type, type) or not issubclass(state_type, CspEnum):
                return None
        elif isinstance(slice_node, ast.Attribute):
            try:
                state_type = eval(ast.unparse(slice_node), globalns)
                if not isinstance(state_type, type) or not issubclass(state_type, CspEnum):
                    return None
            except Exception:
                return None
        else:
            return None

        if isinstance(node.value, ast.Attribute):
            try:
                initial_value = eval(ast.unparse(node.value), globalns)
                if not isinstance(initial_value, state_type):
                    raise TypeError(
                        f"State '{var_name}' expected {state_type.__name__}, got {type(initial_value).__name__}"
                    )
            except NameError as e:
                raise TypeError(f"Could not resolve initial value for state '{var_name}': {e}")
        else:
            raise TypeError(
                f"State '{var_name}' must be initialized with an enum value (e.g., {state_type.__name__}.VALUE)"
            )

        return StateVariableInfo(var_name, initial_value, state_type)


def _csp_enum_attribute_transformer(node: ast.AST, globalns: dict, variable_factory) -> Optional[ast.AST]:
    """Transform MyEnum.VALUE to its integer constant."""
    enum_member = _resolve_csp_enum_member(node, globalns)
    if enum_member is None:
        return None
    return ast.Constant(value=enum_member.value)


def register():
    """Register CSP enum type support."""
    TypeFactory.register(CspEnumType)
    NumbaTypeInference.register_attr_lowerer(_csp_enum_attribute_transformer)
