import inspect
import types
import typing

from csp.impl.__cspimpl import _cspimpl
from csp.impl.error_handling import ExceptionContext
from csp.impl.mem_cache import csp_memoized_graph_object, function_full_name
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.generic_values_resolver import GenericValuesResolver
from csp.impl.wiring.edge import Edge
from csp.impl.wiring.node import NodeDef, _create_node
from csp.impl.wiring.numba_node_parser import NumbaNodeParser


class NumbaNumbaNodeDefMeta(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instantiate_func = self._instantiate_impl
        if self.memoize or self.force_memoize:
            self._instantiate_func = csp_memoized_graph_object(
                self._instantiate_impl, force_memoize=self.force_memoize, function_name=function_full_name(self._impl)
            )

    def _instantiate_impl(self, __forced_tvars, args, kwargs):
        ## Parse inputs
        inputs, scalars, tvars = self._signature.parse_inputs(__forced_tvars, *args, **kwargs)
        NumbaNodeDef = super().__call__(
            inputs, scalars, self._state_types, tvars, self._impl, pre_create_hook=self._pre_create_hook
        )
        outputs, output_types = self._signature.create_outputs(NumbaNodeDef, tvars)
        NumbaNodeDef._outputs = output_types
        return outputs

    def _instantiate(self, __forced_tvars, *args, **kwargs):
        return self._instantiate_func(__forced_tvars, args=args, kwargs=kwargs)

    def __call__(cls, *args, **kwargs):
        return cls._instantiate(None, *args, **kwargs)

    def using(cls, **__forced_tvars):
        return lambda *args, **kwargs: cls._instantiate(__forced_tvars, *args, **kwargs)

    def __get__(self, instance, owner):
        # This little bit of magic allows nodes to be defined as class members, this intercepts method calls
        # and converts them to bound methods
        if instance is not None:
            return types.MethodType(self, instance)
        return self


##Every NumbaNodeDef instance represents an instance of a wiring-time node
class NumbaNodeDef(NodeDef):
    _COMPILED_NODES_CACHE = {}

    def __init__(self, inputs, scalars, state_types, tvars, impl, pre_create_hook):
        super().__init__(inputs=inputs, scalars=scalars, tvars=tvars, impl=impl, pre_create_hook=pre_create_hook)
        self._state_types = state_types
        self._state_class = None
        self._state_instance = None
        self._init_func = None
        self._next_func = None

    def _create(self, engine, memo):
        from csp.impl.wiring.numba_utils.csp_cpp_numba_interface import NumbaTSTypedFunctionResolver, ffi_ptr_to_int
        from csp.impl.wiring.numba_utils.numba_type_resolver import NumbaTypeResolver

        exposed_utility_values = {}
        cache_key = [self._impl]
        for (ts_idx, basket_idx), input in self.ts_inputs():
            exposed_utility_values[NumbaNodeParser.get_ts_input_value_getter_name(ts_idx)] = (
                NumbaTSTypedFunctionResolver.get_value_getter_function(input.tstype.typ)
            )
            cache_key.append(input.tstype)

        for output in self._outputs:
            assert output.kind.is_single_ts()
            ts_idx = output.ts_idx
            exposed_utility_values[NumbaNodeParser.get_ts_out_value_return_name(ts_idx)] = (
                NumbaTSTypedFunctionResolver.get_value_returner_function(output.typ.typ)
            )

        numba_scalar_types = []
        for i in self._signature.raw_inputs():
            if i.kind.is_scalar():
                resolved_python_type = GenericValuesResolver.resolve_generic_values(i.typ, self._tvars)
                numba_type = NumbaTypeResolver.resolve_numba_type(resolved_python_type)
                exposed_utility_values[NumbaNodeParser.get_arg_type_var_name(i.name)] = numba_type
                cache_key.append(resolved_python_type)
                numba_scalar_types.append(numba_type)

        for var_name, var_type in self._state_types.items():
            resolved_python_type = GenericValuesResolver.resolve_generic_values(var_type, self._tvars)
            numba_type = NumbaTypeResolver.resolve_numba_type(resolved_python_type)
            exposed_utility_values[NumbaNodeParser.get_state_type_var_name(var_name)] = numba_type
            cache_key.append(resolved_python_type)

        cache_key = tuple(cache_key)

        compiled_node_tuple = self._COMPILED_NODES_CACHE.get(cache_key)
        if compiled_node_tuple is None:
            compiled_node_tuple = self._impl(*self._scalars, **exposed_utility_values)
            self._COMPILED_NODES_CACHE[cache_key] = compiled_node_tuple

        self._state_class, self._init_func, self._next_func = compiled_node_tuple

        scalars = [NumbaTypeResolver.transform_scalar(v, t) for v, t in zip(self._scalars, numba_scalar_types)]

        self._state_instance = self._state_class(*scalars) if self._state_class is not None else None
        # TODO: handle the following here:
        #   alarms
        #   baskets

        if len(self._signature.alarms()) != 0:
            raise NotImplementedError("Alarms are not supported in numba nodes")

        # alarms = self._signature._create_alarms(self._tvars)
        # inputs = [alarm.typ.typ for alarm in alarms]
        inputs = []
        for input in self._inputs:
            if isinstance(input, Edge):
                inputs.append(input.tstype.typ)
            else:
                raise NotImplementedError(f"input of type {type(input)} is not supported")
            # elif isinstance(input, list):
            #     inputs.append((len(input), input[0].tstype.typ))
            # else:  # dict
            #     assert (isinstance(input, dict))
            #     inputs.append((list(input.keys()), next(iter(input.values())).tstype.typ))

        inputs = tuple(inputs)  # tuple(x.tstype.typ if isinstance( x, Edge ) else ( len( x ),  for x in self._inputs)
        outputs = self._outputs

        if not isinstance(outputs, tuple):
            outputs = tuple([outputs])

        outputs = tuple(ContainerTypeNormalizer.normalized_type_to_actual_python_type(odef.typ.typ) for odef in outputs)

        init_callback_ptr_as_int = ffi_ptr_to_int(self._init_func.cffi)
        callback_ptr_as_int = ffi_ptr_to_int(self._next_func.cffi)
        state_ptr_as_int = self._state_instance.get_ptr() if self._state_instance is not None else 0

        # We need the cpp to hold reference to all the relevant python objects so that they don't get garbage
        # collected
        data_reference = (self._init_func, self._next_func, self._state_instance, self._state_class)

        node = _cspimpl.PyNumbaNode(
            engine, inputs, outputs, state_ptr_as_int, init_callback_ptr_as_int, callback_ptr_as_int, data_reference
        )

        for idx, output_type in enumerate(outputs):
            node.create_output(idx, output_type)

        # for idx, alarm in enumerate(alarms):
        #     node.create_alarm(idx, alarm.typ.typ)
        return node


# The decorator
def numba_node(
    func=None,
    *,
    memoize=True,
    force_memoize=False,
    debug_print=False,
    debug_print_python=False,
    state_types: typing.Union[typing.Dict[str, typing.Any], None] = None,
    pre_create_hook=None,
):
    """
    :param func:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param debug_print:
    :param debug_print_python:
    :param state_types:
    :return:
    """
    raise NotImplementedError("numba nodes not yet supported")

    func_frame = inspect.currentframe().f_back

    def _impl(func):
        with ExceptionContext():
            parser = NumbaNodeParser(func.__name__, func, func_frame, debug_print=debug_print, state_types=state_types)
            parser.parse()

            nodetype = NumbaNumbaNodeDefMeta(
                func.__name__,
                (NumbaNodeDef,),
                {
                    "_signature": parser._signature,
                    "_impl": parser._impl,
                    "_state_types": parser._state_types,
                    "memoize": memoize,
                    "force_memoize": force_memoize,
                    "python": _create_node(
                        func=func,
                        func_frame=func_frame,
                        debug_print=debug_print_python,
                        memoize=memoize,
                        force_memoize=force_memoize,
                        cppimpl=None,
                        pre_create_hook=pre_create_hook,
                    ),
                    "_pre_create_hook": pre_create_hook,
                    "__wrapped__": func,
                    "__doc__": parser._docstring,
                },
            )
            return nodetype

    if func is None:
        return _impl
    else:
        return _impl(func)
