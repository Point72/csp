import inspect
from typing import Callable, Optional, get_args, get_origin

from numba_cfunc_compiler.compilation_context import CompilationContext
from numba_cfunc_compiler.defaults import register_types as register_default_types
from numba_cfunc_compiler.defaults.source_categories import (
    ConstantCategory,
    LifecycleCategory,
    OutputCategory,
    StateCategory,
)
from numba_cfunc_compiler.numba_config import NumbaDict, NumbaList
from numba_cfunc_compiler.numba_core import create_compiled_func
from numba_cfunc_compiler.source_registry import SourceRegistry

from csp.impl.__cspimpl import _cspimpl
from csp.impl.wiring.csp_numba.csp_node_transformer import CspNodeTransformer
from csp.impl.wiring.csp_numba.enum_support import register as register_enum_support
from csp.impl.wiring.csp_numba.input_handlers import CspInputMetadata, register as register_input_handlers
from csp.impl.wiring.csp_numba.output_handlers import register as register_output_handlers
from csp.impl.wiring.csp_numba.signal_set_support import (
    register_ast_handlers as register_signal_set_ast_handlers,
)
from csp.impl.wiring.csp_numba.signal_support import SignalCategory
from csp.impl.wiring.csp_numba.struct_support import (
    register as register_struct_support,
    struct_enum_store,
    struct_enum_value,
)
from csp.impl.wiring.node import NodeDef, NodeDefMeta
from csp.impl.wiring.node_parser import NodeDefinitionParser

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


class _NumbaNodeDefinitionParser(NodeDefinitionParser):
    def parse_func_signature(self, funcdef):
        inputs, defaults, outputs = super().parse_func_signature(funcdef)
        normalized_inputs = []
        for input_def in inputs:
            origin = get_origin(input_def.typ)
            if origin is NumbaList:
                input_def = input_def._replace(typ=list[get_args(input_def.typ)[0]])
            elif origin is NumbaDict:
                input_def = input_def._replace(typ=dict[get_args(input_def.typ)])
            normalized_inputs.append(input_def)
        return normalized_inputs, defaults, outputs


class NumbaNodeDefMeta(NodeDefMeta):
    def _instantiate_impl(self, __forced_tvars, name, args, kwargs):
        inputs, scalars, tvars = self._signature.parse_inputs(__forced_tvars, *args, **kwargs)
        input_values = iter(inputs)
        scalar_values = iter(scalars)
        compiler_kwargs = {}
        for input_def in self._signature.inputs:
            if input_def.kind.is_any_ts():
                value = next(input_values)
                compiler_kwargs[input_def.name] = dict(enumerate(value)) if isinstance(value, (list, tuple)) else value
            else:
                compiler_kwargs[input_def.name] = next(scalar_values)

        compilation = self._compile_call(**compiler_kwargs)
        flattened_inputs = tuple(compilation.ordered_input_signals)
        nodedef = type.__call__(self, flattened_inputs, scalars, tvars, self._impl, self._pre_create_hook, compilation)
        return self._finalize_nodedef(nodedef, inputs, scalars, tvars, name)


class NumbaNodeDef(NodeDef):
    def __init__(self, inputs, scalars, tvars, impl, pre_create_hook, compilation):
        super().__init__(inputs, scalars, tvars, impl, pre_create_hook)
        self._compilation = compilation

    def _create(self, engine, memo):
        if self._pre_create_hook:
            self._pre_create_hook(engine, memo)

        compiled_func = self._compilation.compiled_func
        ordered_inputs = tuple(edge for _, edge in self.ts_inputs())

        node = _cspimpl.PyNumbaNode(
            engine,
            compiled_func.address,
            ordered_inputs,
            self._output_types,
            self._compilation.state_values,
            self._compilation.nrt_state_indices,
            self._compilation.struct_state_indices,
            self._compilation.struct_state_sizes,
            compiled_func,
        )

        self._create_outputs(node)
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

    func_frame = inspect.currentframe().f_back

    def _impl(func: Callable) -> Callable:
        node_name = name or func.__name__
        parser = _NumbaNodeDefinitionParser(node_name, func, func_frame)
        parser.parse()
        definition = parser.definition
        if definition.signature.alarms():
            raise NotImplementedError("numba_node does not support alarms")

        transformer = CspNodeTransformer()
        transformed = transformer.transform_csp_node(definition)
        transformed_ast = transformed.transformed_ast

        transformed_globals = func.__globals__.copy()

        if func.__code__.co_freevars and func.__closure__:
            for var_name, cell in zip(func.__code__.co_freevars, func.__closure__):
                transformed_globals[var_name] = cell.cell_contents

        if globals:
            transformed_globals.update(globals)

        transformed_globals["struct_enum_value"] = struct_enum_value
        transformed_globals["struct_enum_store"] = struct_enum_store

        def get_edge_type(edge) -> type:
            return edge.tstype.typ

        original_signature = inspect.signature(
            func,
            globals=transformed_globals,
            locals=func_frame.f_locals,
            eval_str=True,
        )
        compiler_signature = original_signature.replace(
            parameters=[
                param.replace(
                    annotation=CspInputMetadata(
                        category="signal_set" if definition.signature.input(param.name).kind.is_basket() else "signal"
                    )
                )
                if definition.signature.input(param.name).kind.is_any_ts()
                else param
                for param in original_signature.parameters.values()
            ]
        )

        def compile_call(*args, **kwargs):
            with _get_csp_context():
                return create_compiled_func(
                    transformed_ast,
                    *args,
                    extract_python_type_fn=get_edge_type,
                    decorator_name="@numba_node",
                    func_globals=transformed_globals,
                    signature=compiler_signature,
                    call_globals=transformed_globals,
                    start_body=transformed.start_body,
                    stop_body=transformed.stop_body,
                    **kwargs,
                )

        nodetype = NumbaNodeDefMeta(
            node_name,
            (NumbaNodeDef,),
            {
                "_signature": definition.signature,
                "_impl": func,
                "_compile_call": staticmethod(compile_call),
                "memoize": True,
                "force_memoize": False,
                "_cppimpl": None,
                "_pre_create_hook": None,
                "__wrapped__": func,
                "__module__": func.__module__,
                "__doc__": func.__doc__,
                "_numba_transformed": transformed,
                "_numba_transformed_ast": transformed_ast,
            },
        )
        return nodetype

    if func is None:
        return _impl
    else:
        return _impl(func)
