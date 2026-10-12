"""CSP-owned Numba view of native structs, including presence and enum fields."""

import ctypes
import hashlib
from dataclasses import dataclass
from functools import cache

from llvmlite import binding as llvm, ir
from numba import types
from numba.core.errors import TypingError
from numba.core.typing.templates import AttributeTemplate
from numba.extending import (
    infer_getattr,
    intrinsic,
    lower_getattr_generic,
    lower_setattr_generic,
    models,
    register_model,
)
from numba_cfunc_compiler.types.builtin.struct.native import StructPtrType
from numba_cfunc_compiler.types.native.ffi import declare_symbol
from numba_cfunc_compiler.types.native.state import StatePayload
from numba_cfunc_compiler.types.policy import ValueSemantics

from csp.impl.__cspimpl import _cspimpl
from csp.impl.types.typing_utils import CspTypingUtils
from csp.impl.wiring.csp_numba.enum_support import register_family

_PRIMITIVES = {
    "BOOL": types.uint8,
    "INT8": types.int8,
    "UINT8": types.uint8,
    "INT16": types.int16,
    "UINT16": types.uint16,
    "INT32": types.int32,
    "UINT32": types.uint32,
    "INT64": types.int64,
    "UINT64": types.uint64,
    "DOUBLE": types.float64,
    "FLOAT": types.float32,
}

_LIBRARY = None
_SYMBOLS = (
    "csp_numba_struct_enum_field_value",
    "csp_numba_struct_enum_field_set",
)


def register_symbols():
    """Make CSP's exported native helpers available to Numba's JIT linker."""
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(_cspimpl.__file__)
        for name in _SYMBOLS:
            address = ctypes.cast(getattr(_LIBRARY, name), ctypes.c_void_p).value
            llvm.add_symbol(name, address)


@dataclass(frozen=True)
class CspField:
    name: str
    offset: int
    size: int
    mask_offset: int
    mask_bitmask: int
    numba_type: types.Type
    is_enum: bool

    @property
    def key(self):
        return (
            self.name,
            self.offset,
            self.size,
            self.mask_offset,
            self.mask_bitmask,
            self.numba_type.name,
            self.is_enum,
        )


def _fields(py_type, metadata):
    fields = []
    optional_fields = (
        {
            name
            for name, annotation in py_type.metadata(typed=True).items()
            if CspTypingUtils.is_optional_type(annotation)
        }
        if py_type.is_strict()
        else set()
    )
    for item in metadata.get("fields", ()):
        name = item["fieldname"]
        if name in optional_fields:
            raise TypeError(
                f"CSP struct field '{name}' is optional; optional native fields are not supported by numba_node"
            )
        kind = item["type"].get("type", "STRUCT")
        is_enum = kind == "ENUM"
        if is_enum:
            numba_type = register_family(item["type"]["pytype"]).value_type
        else:
            numba_type = _PRIMITIVES.get(kind)
            if numba_type is None:
                raise TypeError(f"CSP struct field '{name}' has unsupported native type '{kind}'")
            if item["size"] != numba_type.bitwidth // 8:
                raise ValueError(f"CSP struct field '{name}' has unexpected size {item['size']}")
        fields.append(
            CspField(
                name,
                item["offset"],
                item["size"],
                item["mask_offset"],
                item["mask_bitmask"],
                numba_type,
                is_enum,
            )
        )
    return tuple(fields)


class CspStructViewType(StructPtrType):
    """Nominal pointer to a CSP-owned native struct."""

    def __init__(self, py_type, metadata):
        self.python_type = py_type  # Retain the PyStructMeta used by native helpers.
        self.fields = _fields(py_type, metadata)
        self.host_size = metadata["size"]
        self.alignment = max((ctypes.alignment(ctypes.c_void_p), *(item["alignment"] for item in metadata["fields"])))
        self.schema_key = ("csp-struct-view-v1", id(py_type), self.host_size, tuple(field.key for field in self.fields))
        fingerprint = hashlib.sha256(repr(self.schema_key).encode()).hexdigest()[:16]
        # The compiler copies StructPtrType outputs by value. CSP supplies its own
        # field accessors, so this view has a CSP layout rather than StructLayout.
        types.Type.__init__(self, f"CspStructView[{py_type.__module__}.{py_type.__qualname__}:{fingerprint}]")

    @property
    def key(self):
        return self.python_type, self.schema_key

    def get_field(self, name):
        return next((field for field in self.fields if field.name == name), None)


@register_model(CspStructViewType)
class CspStructViewModel(models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, ir.IntType(8).as_pointer())


@cache
def csp_struct_view_type(py_type):
    metadata = py_type._metadata_info()
    if not metadata.get("is_native", False):
        raise TypeError(f"numba_node only supports native csp.Struct types; '{py_type.__name__}' is non-native")
    return CspStructViewType(py_type, metadata)


@cache
def csp_struct_view(py_type):
    view_type = csp_struct_view_type(py_type)

    @intrinsic
    def view(typingctx, raw_ptr):
        if raw_ptr != types.voidptr:
            return None
        sig = view_type(raw_ptr)

        def codegen(context, builder, signature, args):
            return builder.bitcast(args[0], context.get_value_type(view_type))

        return sig, codegen

    return view


@cache
def _reject_replacement(py_type, name):
    view_type = csp_struct_view_type(py_type)

    @intrinsic
    def store(typingctx, slot, rhs):
        raise TypingError(f"State[{view_type.python_type.__name__}] '{name}' does not support whole-struct replacement")

    return store


@cache
def csp_struct_payload(py_type):
    view_type = csp_struct_view_type(py_type)
    return StatePayload(
        label=f"{py_type.__module__}.{py_type.__qualname__}",
        semantics=ValueSemantics.BORROWED_VIEW,
        native_type=view_type,
        host_size=view_type.host_size,
        alignment=view_type.alignment,
        layout_key=view_type.schema_key,
        replacement=False,
        bind_function=csp_struct_view(py_type),
        store_factory=lambda name: _reject_replacement(py_type, name),
        mutable=True,
    )


@infer_getattr
class CspStructAttributes(AttributeTemplate):
    key = CspStructViewType

    def generic_resolve(self, typ, attr):
        field = typ.get_field(attr)
        return field.numba_type if field is not None else None


def _type_pointer(context, builder, typ):
    return builder.inttoptr(context.get_constant(types.intp, id(typ.python_type)), ir.IntType(8).as_pointer())


def _field_offset(context, field):
    return context.get_constant(types.int64, field.offset)


def _byte_pointer(context, builder, value, offset):
    return builder.gep(value, [context.get_constant(types.intp, offset)])


def _return_failure(context, builder):
    # CSP compiles its callbacks with an int8 status result: 1 succeeds, 0 fails.
    context.call_conv.return_value(builder, ir.Constant(ir.IntType(8), 0))


def _guard_struct_pointer(context, builder, value):
    null = builder.icmp_unsigned("==", value, ir.Constant(value.type, None))
    with builder.if_then(null, likely=False):
        _return_failure(context, builder)


def _guard_field_read(context, builder, value, field):
    _guard_struct_pointer(context, builder, value)
    mask = builder.load(_byte_pointer(context, builder, value, field.mask_offset), align=1)
    unset = builder.icmp_unsigned(
        "==", builder.and_(mask, ir.Constant(ir.IntType(8), field.mask_bitmask)), ir.Constant(ir.IntType(8), 0)
    )
    with builder.if_then(unset, likely=False):
        _return_failure(context, builder)


@lower_getattr_generic(CspStructViewType)
def csp_struct_getattr(context, builder, typ, value, attr):
    field = typ.get_field(attr)
    if field is None:
        raise AttributeError(attr)
    _guard_field_read(context, builder, value, field)
    if field.is_enum:
        native_sig = ir.FunctionType(ir.IntType(64), [ir.IntType(8).as_pointer(), ir.IntType(64)])
        function = declare_symbol(builder, "csp_numba_struct_enum_field_value", native_sig)
        return builder.call(function, [value, _field_offset(context, field)])

    result_type = context.get_value_type(field.numba_type)
    pointer = builder.bitcast(_byte_pointer(context, builder, value, field.offset), result_type.as_pointer())
    return builder.load(pointer, align=1)


@lower_setattr_generic(CspStructViewType)
def csp_struct_setattr(context, builder, sig, args, attr):
    typ, value_type = sig.args
    field = typ.get_field(attr)
    if field is None:
        raise AttributeError(attr)
    value = context.cast(builder, args[1], value_type, field.numba_type)
    _guard_struct_pointer(context, builder, args[0])
    if field.is_enum:
        native_sig = ir.FunctionType(
            ir.IntType(8), [ir.IntType(8).as_pointer(), ir.IntType(64), ir.IntType(64), ir.IntType(8).as_pointer()]
        )
        function = declare_symbol(builder, "csp_numba_struct_enum_field_set", native_sig)
        success = builder.call(
            function, [args[0], _field_offset(context, field), value, _type_pointer(context, builder, typ)]
        )
        with builder.if_then(builder.icmp_unsigned("==", success, ir.Constant(ir.IntType(8), 0)), likely=False):
            _return_failure(context, builder)
        return

    pointer = builder.bitcast(
        _byte_pointer(context, builder, args[0], field.offset),
        context.get_value_type(field.numba_type).as_pointer(),
    )
    builder.store(value, pointer, align=1)
    mask_pointer = _byte_pointer(context, builder, args[0], field.mask_offset)
    mask = builder.load(mask_pointer, align=1)
    builder.store(builder.or_(mask, ir.Constant(ir.IntType(8), field.mask_bitmask)), mask_pointer, align=1)
