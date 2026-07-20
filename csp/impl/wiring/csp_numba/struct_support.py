import ast
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Optional

from llvmlite import ir
from numba import types as numba_types
from numba.extending import intrinsic

from csp.impl.struct import Struct as CspStruct

from numba_type_utils.type_factory import TypeFactory
from numba_type_utils.models import StateVariableInfo
from numba_type_utils.defaults.struct_support import StructType, StructFieldInfo
from numba_type_utils.numba_config import NumbaTypeRegistry
from numba_type_utils.numba_type_inference import NumbaTypeInference
from numba_type_utils.defaults.struct_support import struct_attribute_transformer, struct_attr_handler
from numba_type_utils.utils.ast import AST


def _is_csp_struct_subclass(value: Any) -> bool:
    return isinstance(value, type) and issubclass(value, CspStruct) and value is not CspStruct


def _require_native_csp_struct_type(var_type: type) -> dict:
    if not _is_csp_struct_subclass(var_type):
        raise TypeError(f"{var_type} is not a csp.Struct subclass")

    metadata_info = var_type._metadata_info()
    if not metadata_info.get("is_native", False):
        raise TypeError(f"numba_node only supports native csp.Struct types; '{var_type.__name__}' is non-native")
    return metadata_info


@intrinsic
def struct_enum_value(typingctx, struct_ptr, field_offset_const):
    """Read a CspEnum struct field and return its integer value."""
    if struct_ptr == numba_types.voidptr and isinstance(field_offset_const, numba_types.Literal):
        field_offset = field_offset_const.literal_value
        sig = numba_types.int64(struct_ptr, field_offset_const)

        def codegen(context, builder, signature, args):
            [struct_ptr_val, _field_offset] = args

            i8p = ir.IntType(8).as_pointer()
            i64 = ir.IntType(64)
            module = builder.module
            fn_name = "csp_numba_struct_enum_field_value"
            fn = module.globals.get(fn_name)
            if fn is None:
                fn_ty = ir.FunctionType(i64, [i8p, i64])
                fn = ir.Function(module, fn_ty, name=fn_name)
                fn.attributes.add("readonly")
                fn.attributes.add("nounwind")

            struct_bytes = builder.bitcast(struct_ptr_val, i8p)
            offset_val = context.get_constant(numba_types.int64, field_offset)
            return builder.call(fn, [struct_bytes, offset_val])

        return sig, codegen


@intrinsic
def struct_enum_store(typingctx, struct_ptr, field_offset_const, value):
    """Store an integer enum value into a CspEnum struct field."""
    if struct_ptr == numba_types.voidptr and isinstance(field_offset_const, numba_types.Literal):
        field_offset = field_offset_const.literal_value
        sig = numba_types.void(struct_ptr, field_offset_const, numba_types.int64)

        def codegen(context, builder, signature, args):
            [struct_ptr_val, _field_offset, enum_value] = args

            i8p = ir.IntType(8).as_pointer()
            i64 = ir.IntType(64)
            module = builder.module
            fn_name = "csp_numba_struct_enum_field_set"
            fn = module.globals.get(fn_name)
            if fn is None:
                fn_ty = ir.FunctionType(ir.VoidType(), [i8p, i64, i64])
                fn = ir.Function(module, fn_ty, name=fn_name)
                fn.attributes.add("nounwind")

            struct_bytes = builder.bitcast(struct_ptr_val, i8p)
            offset_val = context.get_constant(numba_types.int64, field_offset)
            builder.call(fn, [struct_bytes, offset_val, enum_value])
            return context.get_dummy_value()

        return sig, codegen


@dataclass(frozen=True)
class CspStructType(StructType):
    """StructType for csp.Struct subclasses."""

    _fields_cache: ClassVar[Dict[type, Dict[str, StructFieldInfo]]] = {}
    _field_cpp_types_cache: ClassVar[Dict[type, Dict[str, str]]] = {}
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
        cpp_types = {}

        for field in metadata_info.get("fields", []):
            fields[field["fieldname"]] = StructFieldInfo(
                name=field["fieldname"],
                offset=field["offset"],
                numba_type_name=NumbaTypeRegistry.cpp_type_to_numba_name(field["type"].get("type", "STRUCT")),
                size=field["size"],
            )
            cpp_types[field["fieldname"]] = field["type"].get("type", "STRUCT")

        cls._fields_cache[var_type] = fields
        cls._field_cpp_types_cache[var_type] = cpp_types
        return fields

    @classmethod
    def _get_struct_size(cls, var_type: type) -> int:
        if var_type in cls._size_cache:
            return cls._size_cache[var_type]

        size = _require_native_csp_struct_type(var_type).get("size", 0)
        cls._size_cache[var_type] = size
        return size

    def _get_field_cpp_type(self, field_name: str) -> Optional[str]:
        struct_type = self.value
        if struct_type is None:
            return None
        if struct_type not in self._field_cpp_types_cache:
            self._get_struct_fields(struct_type)
        return self._field_cpp_types_cache.get(struct_type, {}).get(field_name)

    def get_field(self, struct_expr, field_name: str) -> ast.AST:
        if self._get_field_cpp_type(field_name) != "ENUM":
            return super().get_field(struct_expr, field_name)

        # Native struct enum fields are stored as CspEnum objects, not raw int64
        # values, so the generic byte-offset load path does not work for them.
        field_info = self._get_field_info(field_name)
        struct_ptr = ast.Name(id=struct_expr, ctx=ast.Load()) if isinstance(struct_expr, str) else struct_expr
        return AST.function_call("struct_enum_value", struct_ptr, ast.Constant(value=field_info.offset))

    def set_field(self, struct_name: str, field_name: str, value_expr: ast.AST) -> ast.AST:
        if self._get_field_cpp_type(field_name) != "ENUM":
            return super().set_field(struct_name, field_name, value_expr)

        # Writes need the same special casing so int64 enum values are
        # reconstructed back into the field's native CspEnum representation.
        field_info = self._get_field_info(field_name)
        struct_ptr = ast.Name(id=struct_name, ctx=ast.Load())
        return AST.function_call("struct_enum_store", struct_ptr, ast.Constant(value=field_info.offset), value_expr)

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
    import ctypes

    import llvmlite.binding as llvm
    from csp.impl.__cspimpl import _cspimpl

    TypeFactory.register(CspStructType)
    NumbaTypeInference.register_attr_lowerer(struct_attribute_transformer)
    NumbaTypeInference.register_attr_accessor(struct_attr_handler)

    cdll = ctypes.CDLL(_cspimpl.__file__)
    for symbol_name in (
        "csp_numba_struct_enum_field_value",
        "csp_numba_struct_enum_field_set",
    ):
        fn = getattr(cdll, symbol_name)
        llvm.add_symbol(symbol_name, ctypes.cast(fn, ctypes.c_void_p).value)
