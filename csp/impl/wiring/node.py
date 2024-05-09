import inspect
import logging
import types

from csp.impl.__cspimpl import _cspimpl
from csp.impl.error_handling import ExceptionContext
from csp.impl.mem_cache import csp_memoized_graph_object, function_full_name
from csp.impl.types.common_definitions import ArgKind
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.wiring import tstype
from csp.impl.wiring.edge import Edge
from csp.impl.wiring.node_parser import NodeParser


##Every NodeDefMeta instance represents a @csp.node definition
class NodeDefMeta(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instantiate_func = self._instantiate_impl

        if self.memoize or self.force_memoize:
            self._instantiate_func = csp_memoized_graph_object(
                self._instantiate_impl, force_memoize=self.force_memoize, function_name=function_full_name(self._impl)
            )

        if self._cppimpl:
            # expose ability top call python impl directly
            python_kwargs = args[2].copy()
            python_kwargs["_cppimpl"] = None
            self.python = NodeDefMeta(self.__name__, self.__bases__, python_kwargs)

    def _instantiate_impl(self, __forced_tvars, name, args, kwargs):
        ## Parse inputs
        inputs, scalars, tvars = self._signature.parse_inputs(__forced_tvars, *args, **kwargs)
        nodedef = super().__call__(inputs, scalars, tvars, self._impl, self._pre_create_hook)

        basket_shape_eval_inputs = list(scalars)
        for input in inputs:
            if isinstance(input, list) or isinstance(input, dict):
                basket_shape_eval_inputs.append(input)

        outputs, output_types = self._signature.create_outputs(nodedef, tvars, basket_shape_eval_inputs)
        nodedef.outputs = output_types
        nodedef.__name__ = name if name else self.__name__
        return outputs

    def _instantiate(self, __forced_tvars, name=None, *args, **kwargs):
        return self._instantiate_func(__forced_tvars, name=name, args=args, kwargs=kwargs)

    def __call__(self, *args, **kwargs):
        return self._instantiate(None, None, *args, **kwargs)

    def using(cls, name=None, **__forced_tvars):
        return lambda *args, **kwargs: cls._instantiate(__forced_tvars, name, *args, **kwargs)

    def __get__(self, instance, owner):
        # This little bit of magic allows nodes to be defined as class members, this intercepts method calls
        # and converts them to bound methods
        if instance is not None:
            return types.MethodType(self, instance)
        return self


##Every NodeDef instance represents an instance of a wiring-time node
class NodeDef:
    def __init__(self, inputs, scalars, tvars, impl, pre_create_hook):
        self._inputs = inputs
        self._scalars = scalars
        self._impl = impl
        self._tvars = tvars
        self._outputs = None
        self._output_types = None
        self._pre_create_hook = pre_create_hook

    @property
    def _signature(self):
        return type(self)._signature

    @property
    def outputs(self):
        return self._outputs

    @classmethod
    def _get_normalized_ts_or_basket_type(cls, output):
        if output.shape:
            return output.shape, ContainerTypeNormalizer.normalized_type_to_actual_python_type(output.typ.typ)
        elif output.kind == ArgKind.DYNAMIC_BASKET_TS:
            # Type returned is type of basket element
            return None, ContainerTypeNormalizer.normalized_type_to_actual_python_type(output.typ.__args__[1].typ)
        else:
            return ContainerTypeNormalizer.normalized_type_to_actual_python_type(output.typ.typ)

    @classmethod
    def _normalize_inputs_to_cpp_types(cls, inputs, name):
        """convert input edges into the c++ equivalents expected for construction"""
        converted = []
        for input in inputs:
            if isinstance(input, Edge):
                if tstype.isTsDynamicBasket(input.tstype):
                    converted.append(
                        (
                            None,
                            ContainerTypeNormalizer.normalized_type_to_actual_python_type(input.tstype.__args__[1].typ),
                        )
                    )
                else:
                    converted.append(ContainerTypeNormalizer.normalized_type_to_actual_python_type(input.tstype.typ))
            elif isinstance(input, (list, tuple)):
                converted.append(
                    (len(input), ContainerTypeNormalizer.normalized_type_to_actual_python_type(input[0].tstype.typ))
                )
            else:  # dict
                if not (isinstance(input, dict)):
                    raise ValueError(f"Unexpected value {input} passed to {name}")
                converted.append(
                    (
                        list(input.keys()),
                        ContainerTypeNormalizer.normalized_type_to_actual_python_type(
                            next(iter(input.values())).tstype.typ
                        ),
                    )
                )

        return tuple(converted)

    @outputs.setter
    def outputs(self, value):
        self._outputs = value
        self._output_types = tuple(self._get_normalized_ts_or_basket_type(output) for output in self._outputs)

    def ts_inputs(self):
        """return generator of proper (( ts_idx, basket_idx) ,edge) inputs adjusted for alarms"""

        # for idx, input in enumerate( self._inputs ):
        #    yield ( num_alarms + idx, input )
        ts_idx = len(self._signature.alarms())
        for input in self._inputs:
            if isinstance(input, list):
                for basket_idx, basket_input in enumerate(input):
                    yield ((ts_idx, basket_idx), basket_input)
            elif isinstance(input, dict):
                for basket_idx, basket_input in enumerate(input.values()):
                    yield ((ts_idx, basket_idx), basket_input)
            else:
                yield ((ts_idx, -1), input)

            ts_idx += 1

    def _create(self, engine, memo):
        if self._pre_create_hook:
            self._pre_create_hook(engine, memo)
        alarms = self._signature._create_alarms(self._tvars)
        inputs = tuple(ContainerTypeNormalizer.normalized_type_to_actual_python_type(alarm.typ.typ) for alarm in alarms)
        inputs = inputs + self._normalize_inputs_to_cpp_types(self._inputs, self._signature._name)

        node = None
        if self._cppimpl:
            cppinputs = [
                (input_def.name, input_type, input_def.ts_idx, input_def.kind == ArgKind.ALARM)
                for input_type, input_def in zip(inputs, self._signature.ts_inputs)
            ]
            cppoutputs = [
                (output_def.name if output_def.name else "", output_type, output_def.ts_idx)
                for output_type, output_def in zip(self._output_types, self._signature._outputs)
            ]

            tsdefs = (tuple(cppinputs), tuple(cppoutputs))
            cppscalars = {scalar_def.name: value for scalar_def, value in zip(self._signature.scalars, self._scalars)}
            try:
                node = self._cppimpl(self._signature._name, engine, *tsdefs, cppscalars)
            except NotImplementedError as err:
                # Some configurations of some impls may not be supported, so fallback to python
                logging.debug(
                    f'cppimpl for node "{self._signature._name}" doesnt supported the given inputs, defaulting to python impl ({err})'
                )

        if node is None:
            gen = self._impl(*self._scalars)
            node = _cspimpl.PyNode(engine, inputs, self._output_types, gen)

        for idx, output_type in enumerate(self._output_types):
            node.create_output(idx, output_type)

        for idx, alarm in enumerate(alarms):
            node.create_alarm(idx, ContainerTypeNormalizer.normalized_type_to_actual_python_type(alarm.typ.typ))
        return node


def _create_node(
    func,
    func_frame,
    debug_print,
    memoize,
    force_memoize,
    cppimpl,
    pre_create_hook,
    name,
):
    parser = NodeParser(
        name,
        func,
        func_frame,
        debug_print=debug_print,
    )
    parser.parse()

    nodetype = NodeDefMeta(
        name,
        (NodeDef,),
        {
            "_signature": parser._signature,
            "_impl": parser._impl,
            "memoize": memoize,
            "force_memoize": force_memoize,
            "_cppimpl": cppimpl,
            "_pre_create_hook": pre_create_hook,
            "__wrapped__": func,
            "__module__": func.__module__,
            "__doc__": parser._docstring,
        },
    )
    return nodetype


def _node_internal_use(
    func=None,
    *,
    func_frame=None,
    memoize=True,
    force_memoize=False,
    debug_print=False,
    cppimpl=None,
    pre_create_hook=None,
    name=None,
):
    """A decorator similar to the @node decorator that exposes some internal arguments that shoudn't be visible to users"""
    func_frame = func_frame if func_frame else inspect.currentframe().f_back

    def _impl(func):
        with ExceptionContext():
            return _create_node(
                func=func,
                func_frame=func_frame,
                debug_print=debug_print,
                memoize=memoize,
                force_memoize=force_memoize,
                cppimpl=cppimpl,
                pre_create_hook=pre_create_hook,
                name=name or func.__name__,
            )

    if func is None:
        return _impl
    else:
        with ExceptionContext():
            return _impl(func)


# The decorator
def node(
    func=None,
    *,
    memoize=True,
    force_memoize=False,
    debug_print=False,
    cppimpl=None,
    name=None,
):
    """
    :param func: The wrapped node function
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param debug_print: A boolean that specifies that processed function should be printed
    :param cppimpl:
    :param name: Provide a custom name for the constructed node type, helpful when viewing a graph with many same-named nodes
    :return:
    """
    with ExceptionContext():
        return _node_internal_use(
            func=func,
            func_frame=inspect.currentframe().f_back,
            memoize=memoize,
            force_memoize=force_memoize,
            debug_print=debug_print,
            cppimpl=cppimpl,
            pre_create_hook=None,
            name=name,
        )
