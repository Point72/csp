import ast
from typing import Optional

from numba_type_utils.utils.ast import AST
from numba_type_utils.method_factory import MethodBase
from numba_type_utils.variable_factory import VariableSource
from numba_type_utils.models import UnknownType, UnknownNumbaType

from csp.impl.wiring.csp_numba.signal_support import (
    INPUTS_ARRAY_NAME,
    TICKED_ARRAY_NAME,
    VALID_ARRAY_NAME,
)

__all__ = [
    "BasketTicked",
    "BasketValid",
    "SignalSetSource",
    "DynamicSignalAccess",
    "register_ast_handlers",
]


class BasketTicked(MethodBase):
    """Check if any signal in a SignalSet has ticked."""

    @classmethod
    def get_name(cls) -> str:
        return "ticked"

    @staticmethod
    def handle(var, args):
        """
        Create: input_ticked[start] != 0 or input_ticked[start+1] != 0 or ...
        """
        if var.length == 0:
            return ast.Constant(value=False)

        comparisons = []
        for i in range(var.length):
            idx = var.start_idx + i
            comp = ast.Compare(
                left=AST.array_access(TICKED_ARRAY_NAME, idx),
                ops=[ast.NotEq()],
                comparators=[ast.Constant(value=0)],
            )
            comparisons.append(comp)

        if len(comparisons) == 1:
            return comparisons[0]

        result = comparisons[0]
        for comp in comparisons[1:]:
            result = ast.BoolOp(op=ast.Or(), values=[result, comp])
        return result


class BasketValid(MethodBase):
    """Check if all signals in a SignalSet are valid."""

    @classmethod
    def get_name(cls) -> str:
        return "valid"

    @staticmethod
    def handle(var, args):
        """
        Create: input_valid[start] != 0 and input_valid[start+1] != 0 and ...
        """
        if var.length == 0:
            return ast.Constant(value=True)

        comparisons = []
        for i in range(var.length):
            idx = var.start_idx + i
            comp = ast.Compare(
                left=AST.array_access(VALID_ARRAY_NAME, idx),
                ops=[ast.NotEq()],
                comparators=[ast.Constant(value=0)],
            )
            comparisons.append(comp)

        if len(comparisons) == 1:
            return comparisons[0]

        result = comparisons[0]
        for comp in comparisons[1:]:
            result = ast.BoolOp(op=ast.And(), values=[result, comp])
        return result


class SignalSetSource(VariableSource):
    """Represents a SignalSet input. Supports indexing into contained signals."""

    def __init__(
        self,
        name: str,
        key_to_child_name: dict,
        start_idx: int = 0,
        element_type=None,
    ):
        super().__init__(UnknownType(UnknownNumbaType(), None), name, [BasketTicked, BasketValid])
        self.key_to_child_name = key_to_child_name
        self.start_idx = start_idx
        self.length = len(key_to_child_name)
        self.element_type = element_type
        self._idx_to_key = {i: k for i, k in enumerate(sorted(key_to_child_name.keys(), key=str))}

    def local_variable_name(self):
        return self.name

    def get_storage_location(self):
        raise ValueError("SignalSetSource has no direct storage location")

    def read(self):
        return None

    def get(self):
        raise ValueError('SignalSetSource cannot be used as a value; index it like set["key"]')

    def is_opaque_pointer(self) -> bool:
        return True

    def get_key_index(self, key) -> int:
        """Get the array index for a given key (relative to start_idx)."""
        keys = list(self.key_to_child_name.keys())
        if key in keys:
            return keys.index(key)
        raise KeyError(f"SignalSet '{self.name}' has no key '{key}'")

    def create_dynamic_access(self, index_expr, variable_factory=None):
        """Create a DynamicSignalAccess for this SignalSet."""
        return DynamicSignalAccess(self, index_expr, variable_factory=variable_factory)


class DynamicSignalAccess(VariableSource):
    """Represents a dynamically-indexed signal from a SignalSet."""

    def __init__(self, signal_set: SignalSetSource, index_expr: ast.AST, variable_factory=None):
        from csp.impl.wiring.csp_numba.signal_support import Ticked, Valid

        element_type = signal_set.element_type or UnknownType(UnknownNumbaType(), None)
        super().__init__(element_type, f"{signal_set.name}[dyn]", [Valid, Ticked], variable_factory)
        self.signal_set = signal_set
        self.index_expr = index_expr
        self.start_idx = signal_set.start_idx

    def local_variable_name(self):
        return f"{self.signal_set.name}_dyn"

    def get_storage_location(self):
        return INPUTS_ARRAY_NAME

    def get_valid_array_name(self):
        return VALID_ARRAY_NAME

    def get_ticked_array_name(self):
        return TICKED_ARRAY_NAME

    def _get_effective_index(self) -> ast.AST:
        """Return AST for: start_idx + index_expr"""
        if self.start_idx == 0:
            return self.index_expr
        return ast.BinOp(
            left=ast.Constant(value=self.start_idx),
            op=ast.Add(),
            right=self.index_expr,
        )

    def is_opaque_pointer(self) -> bool:
        return True

    def read(self):
        return None

    def get(self):
        """Return AST for accessing the signal value."""
        effective_idx = self._get_effective_index()
        array_access = ast.Subscript(
            value=ast.Name(id=INPUTS_ARRAY_NAME, ctx=ast.Load()),
            slice=effective_idx,
            ctx=ast.Load(),
        )

        if self.type and hasattr(self.type, "get_numba_type_name"):
            dtype_name = self.type.get_numba_type_name()
        else:
            dtype_name = "int64"

        cast_call = ast.Call(
            func=ast.Name(id="cast_voidptr_to_ptr", ctx=ast.Load()),
            args=[array_access, ast.Constant(value=dtype_name)],
            keywords=[],
        )

        if self.type and hasattr(self.type, "is_opaque_pointer") and self.type.is_opaque_pointer():
            return cast_call

        return ast.Subscript(
            value=cast_call,
            slice=ast.Constant(value=0),
            ctx=ast.Load(),
        )


def handle_len_signal_set(converter, node: ast.Call) -> Optional[ast.AST]:
    """
    Handle len(signal_set) -> constant length.

    Transforms calls like `len(my_basket)` into a constant when
    `my_basket` is a SignalSetSource.
    """
    if not isinstance(node.func, ast.Name) or node.func.id != "len":
        return None

    if len(node.args) != 1 or not isinstance(node.args[0], ast.Name):
        return None

    var = converter.variable_factory.from_name(node.args[0].id)
    if var is not None and isinstance(var, SignalSetSource):
        return ast.Constant(value=var.length)

    return None


def handle_signal_set_subscript(converter, node: ast.Subscript) -> Optional[ast.AST]:
    """
    Handle signal_set[key] -> load the signal value.

    Transforms subscript access on SignalSetSource into a loaded signal value.
    """
    if not isinstance(node.value, ast.Name):
        return None

    container = converter.variable_factory.from_name(node.value.id)
    if container is None or not isinstance(container, SignalSetSource):
        return None

    child_var = converter.variable_factory.from_ast(visitor=converter, ast_node=node, statements=[])
    return child_var.get()


def handle_signal_set_iteration(converter, node: ast.For) -> Optional[ast.AST]:
    """
    Transform signal set iteration patterns.

    Transform: for key in signal_set.tickedkeys(): ...
    Into:
        for _iter_i in range(signal_set_len):
            if input_ticked[start_idx + _iter_i] != 0:
                key = _iter_i
                ...
    """
    if not isinstance(node.iter, ast.Call):
        return None
    if not isinstance(node.iter.func, ast.Attribute):
        return None

    method_name = node.iter.func.attr
    if method_name not in ("tickedkeys", "validkeys", "keys"):
        return None

    if not isinstance(node.iter.func.value, ast.Name):
        return None

    signal_set_name = node.iter.func.value.id
    signal_set = converter.variable_factory.from_name(signal_set_name)

    if signal_set is None or not isinstance(signal_set, SignalSetSource):
        return None

    if not isinstance(node.target, ast.Name):
        return None
    loop_var = node.target.id

    if method_name == "tickedkeys":
        filter_array = TICKED_ARRAY_NAME
    elif method_name == "validkeys":
        filter_array = VALID_ARRAY_NAME
    else:
        filter_array = None

    iter_var = f"_iter_{signal_set_name}"
    range_call = ast.Call(
        func=ast.Name(id="range", ctx=ast.Load()),
        args=[ast.Constant(value=signal_set.length)],
        keywords=[],
    )

    transformed_body = []

    key_assign = ast.Assign(
        targets=[ast.Name(id=loop_var, ctx=ast.Store())],
        value=ast.Name(id=iter_var, ctx=ast.Load()),
    )
    transformed_body.append(key_assign)

    for stmt in node.body:
        transformed_stmt = converter.visit(stmt)
        if isinstance(transformed_stmt, list):
            transformed_body.extend(transformed_stmt)
        elif transformed_stmt is not None:
            transformed_body.append(transformed_stmt)

    if filter_array is not None:
        if signal_set.start_idx == 0:
            effective_idx = ast.Name(id=iter_var, ctx=ast.Load())
        else:
            effective_idx = ast.BinOp(
                left=ast.Constant(value=signal_set.start_idx),
                op=ast.Add(),
                right=ast.Name(id=iter_var, ctx=ast.Load()),
            )

        condition = ast.Compare(
            left=ast.Subscript(
                value=ast.Name(id=filter_array, ctx=ast.Load()),
                slice=effective_idx,
                ctx=ast.Load(),
            ),
            ops=[ast.NotEq()],
            comparators=[ast.Constant(value=0)],
        )

        if_stmt = ast.If(
            test=condition,
            body=transformed_body,
            orelse=[],
        )
        final_body = [if_stmt]
    else:
        final_body = transformed_body

    for_loop = ast.For(
        target=ast.Name(id=iter_var, ctx=ast.Store()),
        iter=range_call,
        body=final_body,
        orelse=[],
    )

    ast.fix_missing_locations(for_loop)
    return for_loop


def register_ast_handlers():
    from numba_type_utils.ast_handlers import ASTHandlerRegistry, HandlerPhase

    ASTHandlerRegistry.register("Call", handle_len_signal_set, HandlerPhase.PRE, priority=0)
    ASTHandlerRegistry.register("Subscript", handle_signal_set_subscript, HandlerPhase.PRE, priority=0)
    ASTHandlerRegistry.register("For", handle_signal_set_iteration, HandlerPhase.PRE, priority=0)
