import inspect
import types

from csp.impl.constants import UNSET
from csp.impl.error_handling import ExceptionContext, fmt_errors
from csp.impl.mem_cache import csp_memoized_graph_object, function_full_name
from csp.impl.types.instantiation_type_resolver import GraphOutputTypeResolver
from csp.impl.wiring.graph_parser import GraphParser
from csp.impl.wiring.outputs import OutputsContainer
from csp.impl.wiring.signature import USE_PYDANTIC
from csp.impl.wiring.special_output_names import UNNAMED_OUTPUT_NAME


class _GraphDefMetaUsingAux:
    def __init__(self, graph_meta, _forced_tvars=None):
        self._graph_meta = graph_meta
        self._forced_tvars = _forced_tvars if _forced_tvars else {}

    def using(self, **_forced_tvars):
        new_forced_tvars = {}
        new_forced_tvars.update(self._forced_tvars)
        new_forced_tvars.update(_forced_tvars)
        return _GraphDefMetaUsingAux(self._graph_meta, new_forced_tvars)

    def __call__(self, *args, **kwargs):
        return self._graph_meta._instantiate(self._forced_tvars, *args, **kwargs)


class GraphDefMeta(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instantiate_func = self._instantiate_impl
        if self.memoize or self.force_memoize:
            if self.wrapped_node:
                full_name = function_full_name(self.wrapped_node._impl)
            else:
                full_name = function_full_name(self._func)

            self._instantiate_func = csp_memoized_graph_object(
                self._instantiate_impl, force_memoize=self.force_memoize, function_name=full_name
            )

    def _extract_forced_tvars(cls, d):
        if "_forced_tvars" in d:
            return d.pop("_forced_tvars")
        return {}

    def _instantiate_impl(self, _forced_tvars, signature, args, kwargs):
        inputs, scalars, tvars = signature.parse_inputs(_forced_tvars, *args, allow_none_ts=True, **kwargs)

        basket_shape_eval_inputs = list(scalars)
        for input in inputs:
            if isinstance(input, list) or isinstance(input, dict):
                basket_shape_eval_inputs.append(input)

        expected_outputs = signature.resolve_output_definitions(
            tvars=tvars, basket_shape_eval_inputs=basket_shape_eval_inputs
        )
        res = self._func(*args, **kwargs)

        # Validate graph return values
        if isinstance(res, OutputsContainer):
            outputs_raw = []
            for e_o in expected_outputs:
                output_name = e_o.name if e_o.name else UNNAMED_OUTPUT_NAME
                cur_o = res._get(output_name, UNSET)
                if cur_o is UNSET:
                    raise KeyError(f"Output {output_name} is not returned from the graph")
                outputs_raw.append(cur_o)
            if len(outputs_raw) != len(res):
                # We have some output for some wrong key
                all_output_names = {e_o.name for e_o in expected_outputs}
                for k in res:
                    if k not in all_output_names:
                        raise KeyError(f"Unexpected returned output name {k}")
                raise RuntimeError("Internal error, unexpected code location")
        elif len(expected_outputs) == 1:
            outputs_raw = (res,)
        else:
            if expected_outputs:
                raise ValueError(
                    f'Unexpected return value "{res}" from {self._signature._name}, expected: {expected_outputs}'
                )
            assert res is None

        if res is not None:
            if USE_PYDANTIC:
                from pydantic import ValidationError

                from csp.impl.types.pydantic_type_resolver import TVarValidationContext

                from .signature import OUTPUT_PREFIX

                outputs_dict = {
                    f"{OUTPUT_PREFIX}{out.name}" if out.name else OUTPUT_PREFIX: arg
                    for arg, out in zip(outputs_raw, expected_outputs)
                }
                output_model = self._signature._output_model
                context = TVarValidationContext(
                    forced_tvars=tvars,
                )
                try:
                    _ = output_model.model_validate(outputs_dict, context=context)
                except ValidationError as e:
                    raise TypeError(f"Output type validation error(s).\n{fmt_errors(e, OUTPUT_PREFIX)}") from None
            else:
                _ = GraphOutputTypeResolver(
                    function_name=self._signature._name,
                    output_definitions=expected_outputs,
                    values=outputs_raw,
                    forced_tvars=tvars,
                )
        if signature.special_outputs:
            if expected_outputs[0].name is None:
                res = next(iter(res._values()))
            else:
                res = OutputsContainer(**{k: v for k, v in res._items() if k != -UNNAMED_OUTPUT_NAME})

        return res

    def _instantiate(self, _forced_tvars, *args, **kwargs):
        return self._instantiate_func(_forced_tvars, self._signature, args=args, kwargs=kwargs)

    def __call__(cls, *args, **kwargs):
        return cls._instantiate(None, *args, **kwargs)

    def using(cls, **_forced_tvars):
        return _GraphDefMetaUsingAux(cls, _forced_tvars)

    def __get__(self, instance, owner):
        if instance is not None:
            return types.MethodType(self, instance)
        return self


def _create_graph(
    func_name,
    func_doc,
    impl,
    signature,
    memoize,
    force_memoize,
    wrapped_function=None,
    wrapped_node=None,
):
    return GraphDefMeta(
        func_name,
        (object,),
        {
            "_signature": signature,
            "_func": impl,
            "memoize": memoize,
            "force_memoize": force_memoize,
            "__wrapped__": wrapped_function,
            "__module__": wrapped_function.__module__,
            "wrapped_node": wrapped_node,
            "__doc__": func_doc,
        },
    )


def graph(
    func=None,
    *,
    memoize=True,
    force_memoize=False,
    name=None,
    debug_print=False,
):
    """
    :param func:
    :param memoize: Specify whether the graph should be memoized (default True)
    :param force_memoize: If True, the graph will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param name: Provide a custom name for the constructed graph type
    :param debug_print: A boolean that specifies that processed function should be printed
    :return:
    """
    func_frame = inspect.currentframe().f_back

    def _impl(func):
        with ExceptionContext():
            parser = GraphParser(
                name or func.__name__,
                func,
                func_frame,
                debug_print=debug_print,
            )
            parser.parse()

            signature = parser._signature
            return _create_graph(
                name or func.__name__,
                func.__doc__,
                parser._impl,
                signature,
                memoize,
                force_memoize,
                wrapped_function=func,
            )

    if func is None:
        return _impl
    else:
        with ExceptionContext():
            return _impl(func)
