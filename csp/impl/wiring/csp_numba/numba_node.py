import ast
import functools
import inspect
from typing import Callable, List, Optional, Tuple

from csp.impl.__cspimpl import _cspimpl
from csp.impl.wiring.edge import Edge
from csp.impl.types.tstype import ts
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer

from csp.impl.wiring.csp_numba.csp_node_transformer import CspNodeTransformer
from csp.impl.wiring.csp_numba.input_handlers import register as register_input_handlers
from csp.impl.wiring.csp_numba.output_handlers import register as register_output_handlers
from csp.impl.wiring.csp_numba.enum_support import register as register_enum_support
from csp.impl.wiring.csp_numba.struct_support import register as register_struct_support
from csp.impl.wiring.csp_numba.signal_support import SignalCategory
from csp.impl.wiring.csp_numba.signal_set_support import (
    register_ast_handlers as register_signal_set_ast_handlers,
)

from numba_type_utils.compilation_context import CompilationContext
from numba_type_utils.numba_core import create_compiled_func
from numba_type_utils.numba_config import State, set_output
from numba_type_utils.defaults import register_types as register_default_types
from numba_type_utils.source_registry import SourceRegistry
from numba_type_utils.defaults.source_categories import (
    ConstantCategory,
    OutputCategory,
    StateCategory,
    LifecycleCategory,
)

__all__ = (
    "numba_node",
    "NumbaNodeDef",
)

# Lazily-initialized CompilationContext with all CSP-specific registrations.
# Created once on first use; reused for all subsequent compilations.
_csp_context: CompilationContext | None = None


def _get_csp_context() -> CompilationContext:
    global _csp_context
    if _csp_context is None:
        ctx = CompilationContext()
        with ctx:
            register_default_types()

            SourceRegistry.register(SignalCategory())
            SourceRegistry.register(ConstantCategory())
            SourceRegistry.register(OutputCategory())
            SourceRegistry.register(StateCategory())
            SourceRegistry.register(LifecycleCategory())

            register_input_handlers()
            register_output_handlers()
            register_enum_support()
            register_struct_support()

            register_signal_set_ast_handlers()
        _csp_context = ctx
    return _csp_context


class NumbaNodeDef:
    def __init__(
        self,
        name: str,
        inputs: List[Edge],
        output_types: List[type],
        compiled_func,
        state_values: Tuple,
        nrt_state_indices: Tuple,
        struct_state_indices: Tuple,
        struct_state_sizes: Tuple,
        func_globals: dict,
    ):
        self.__name__ = name
        self._inputs = inputs
        self._output_types = output_types
        self._compiled_func = compiled_func
        self._state_values = state_values
        self._nrt_state_indices = nrt_state_indices
        self._struct_state_indices = struct_state_indices
        self._struct_state_sizes = struct_state_sizes
        self._func_globals = func_globals

    def ts_inputs(self):
        ts_idx = 0
        for edge in self._inputs:
            yield ((ts_idx, -1), edge)
            ts_idx += 1

    def _create(self, engine, memo):
        func_ptr = self._compiled_func.address

        cpp_output_types = tuple(
            ContainerTypeNormalizer.normalized_type_to_actual_python_type(t) for t in self._output_types
        )

        node = _cspimpl.PyNumbaNode(
            engine,
            func_ptr,
            tuple(self._inputs),
            cpp_output_types,
            self._state_values,
            self._nrt_state_indices,
            self._struct_state_indices,
            self._struct_state_sizes,
            self._compiled_func,
        )

        for idx, output_type in enumerate(cpp_output_types):
            node.create_output(idx, output_type)

        return node


def numba_node(
    func: Optional[Callable] = None,
    *,
    name: str = None,
    globals: dict = None,
):
    """
    Decorator that transforms a CSP node function into a Numba-compiled graph node.

    Args:
        func: The function to transform
        name: Custom name for the node type
        globals: Additional globals to make available during compilation
                 (e.g., enum types defined in enclosing scopes)

    Returns:
        A wrapper function that creates the numba node and returns Edge(s)

    Example:
        @numba_node
        def my_node(x: ts[float], y: ts[int]) -> ts[float]:
            with csp.state():
                total = 0.0
            if csp.ticked(x, y):
                total = total + x + y
                return total

    """

    def _impl(func: Callable) -> Callable:
        transformer = CspNodeTransformer()
        transformed = transformer.transform_csp_node(func)
        transformed_ast = ast.parse(transformed.transformed_source).body[0]

        transformed_globals = func.__globals__.copy()

        if func.__code__.co_freevars and func.__closure__:
            for var_name, cell in zip(func.__code__.co_freevars, func.__closure__):
                transformed_globals[var_name] = cell.cell_contents

        if globals:
            transformed_globals.update(globals)

        def get_edge_type(edge: Edge) -> type:
            return edge.tstype.typ

        original_signature = inspect.signature(func)

        @functools.wraps(func)
        def numba_proxy(*args, **kwargs):
            with _get_csp_context():
                result = create_compiled_func(
                    transformed_ast,
                    *args,
                    extract_python_type_fn=get_edge_type,
                    decorator_name="@numba_node",
                    func_globals=transformed_globals,
                    signature=original_signature,
                    call_globals=transformed_globals,
                    start_body=transformed.start_body,
                    stop_body=transformed.stop_body,
                    **kwargs,
                )

            output_names = list(result.named_outputs.keys()) if result.named_outputs else None
            nodedef = NumbaNodeDef(
                name=name or func.__name__,
                inputs=result.ordered_input_signals,
                output_types=result.output_types,
                compiled_func=result.compiled_func,
                state_values=result.state_values,
                nrt_state_indices=result.nrt_state_indices,
                struct_state_indices=result.struct_state_indices,
                struct_state_sizes=result.struct_state_sizes,
                func_globals=transformed_globals,
            )

            if output_names is not None:
                from csp.impl.wiring.outputs import OutputsContainer

                outputs = {}
                for idx, (out_name, out_type) in enumerate(zip(output_names, result.output_types)):
                    edge = Edge(
                        tstype=ts[out_type],
                        nodedef=nodedef,
                        output_idx=idx,
                    )
                    outputs[out_name] = edge
                return OutputsContainer(**outputs)
            else:
                return Edge(
                    tstype=ts[result.output_types[0]],
                    nodedef=nodedef,
                    output_idx=0,
                )

        numba_proxy._numba_transformed = transformed
        numba_proxy._numba_transformed_ast = transformed_ast

        return numba_proxy

    if func is None:
        return _impl
    else:
        return _impl(func)
