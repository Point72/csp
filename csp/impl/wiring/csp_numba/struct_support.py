import ast
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

from csp.impl.struct import Struct as CspStruct

from numba_type_utils.type_factory import TypeFactory
from numba_type_utils.models import StateVariableInfo
from numba_type_utils.defaults.struct_support import StructType, StructFieldInfo
from numba_type_utils.numba_config import NumbaTypeRegistry
from numba_type_utils.numba_type_inference import NumbaTypeInference
from numba_type_utils.defaults.struct_support import struct_attribute_transformer, struct_attr_handler


def _is_csp_struct_subclass(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, CspStruct) and value is not CspStruct


def _require_native_csp_struct_type(var_type: type) -> dict:
    if not _is_csp_struct_subclass(var_type):
        raise TypeError(f"{var_type} is not a csp.Struct subclass")

    metadata_info = var_type._metadata_info()
    if not metadata_info.get("is_native", False):
        raise TypeError(f"numba_node only supports native csp.Struct types; '{var_type.__name__}' is non-native")
    return metadata_info


@dataclass(frozen=True)
class CspStructType(StructType):
    """StructType for csp.Struct subclasses."""

    _fields_cache: ClassVar[Dict[type, Dict[str, StructFieldInfo]]] = {}
    _size_cache: ClassVar[Dict[type, int]] = {}

    @classmethod
    def is_type_supported(cls, var_type: Any) -> bool:
        return _is_csp_struct_subclass(var_type)

    @classmethod
    def _get_struct_fields(cls, var_type: type) -> Dict[str, StructFieldInfo]:
        if var_type in cls._fields_cache:
            return cls._fields_cache[var_type]

        metadata_info = _require_native_csp_struct_type(var_type)
        fields = {}

        for field in metadata_info.get("fields", []):
            fields[field["fieldname"]] = StructFieldInfo(
                name=field["fieldname"],
                offset=field["offset"],
                numba_type_name=NumbaTypeRegistry.cpp_type_to_numba_name(field["type"].get("type", "STRUCT")),
                size=field["size"],
            )

        cls._fields_cache[var_type] = fields
        return fields

    @classmethod
    def _get_struct_size(cls, var_type: type) -> int:
        if var_type in cls._size_cache:
            return cls._size_cache[var_type]

        size = _require_native_csp_struct_type(var_type).get("size", 0)
        cls._size_cache[var_type] = size
        return size

    @classmethod
    def try_parse_state(cls, node: ast.AnnAssign, var_name: str, globalns: dict) -> Optional[StateVariableInfo]:
        """Parse State[MyStruct] = MyStruct() declarations."""
        slice_node = node.annotation.slice

        if isinstance(slice_node, ast.Name):
            state_type = globalns.get(slice_node.id)
            if state_type is None:
                return None
            if not _is_csp_struct_subclass(state_type):
                return None
        elif isinstance(slice_node, ast.Attribute):
            try:
                state_type = eval(ast.unparse(slice_node), globalns)
                if not _is_csp_struct_subclass(state_type):
                    return None
            except Exception:
                return None
        else:
            return None

        _require_native_csp_struct_type(state_type)

        if node.value is None:
            initial_value = state_type()
        elif isinstance(node.value, ast.Call):
            try:
                initial_value = eval(ast.unparse(node.value), globalns)
                if not isinstance(initial_value, state_type):
                    raise TypeError(
                        f"State '{var_name}' expected {state_type.__name__}, got {type(initial_value).__name__}"
                    )
            except NameError as e:
                raise TypeError(f"Could not resolve initial value for state '{var_name}': {e}")
        elif isinstance(node.value, ast.Constant) and node.value.value is None:
            initial_value = state_type()
        else:
            raise TypeError(f"State '{var_name}' must be initialized with a struct constructor or None")

        return StateVariableInfo(var_name, initial_value, state_type)


def register():
    TypeFactory.register(CspStructType)
    NumbaTypeInference.register_attr_lowerer(struct_attribute_transformer)
    NumbaTypeInference.register_attr_accessor(struct_attr_handler)
