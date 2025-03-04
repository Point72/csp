import inspect
from datetime import timedelta
from typing import List

from typing_extensions import override

from csp.impl.__cspimpl import _cspimpl
from csp.impl.mem_cache import csp_memoized_graph_object
from csp.impl.outputadapter import OutputAdapter  # noqa: F401
from csp.impl.types import tstype
from csp.impl.types.common_definitions import ArgKind, InputDef, OutputDef, PushMode, ReplayMode
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.types.tstype import ts
from csp.impl.wiring.signature import Signature

_ = ReplayMode


# Every AdapterDefMeta instance represents an input or output adapter *definition* type
class AdapterDefMeta(type):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instantiate_func = self._instantiate_impl
        if self.memoize or self.force_memoize:
            self._instantiate_func = csp_memoized_graph_object(self._instantiate_impl, force_memoize=self.force_memoize)

    def _instantiate_impl(cls, __forced_tvars, name, args, kwargs):
        inputs, scalars, tvars = cls._signature.parse_inputs(__forced_tvars, *args, **kwargs)
        adapterdef = super().__call__(inputs, scalars, tvars, cls._impl)
        output, output_defs = cls._signature.create_outputs(adapterdef, tvars)
        if name:
            adapterdef.__name__ = name

        if len(output_defs):
            adapterdef.set_output_def(output_defs[0])
            # Note that we augment the returned Edge to be list of expected type, but not the output def
            # output def remains the original type
            if kwargs.get("push_mode", None) == PushMode.BURST:
                output.tstype = tstype.ts[List[output.tstype.typ]]

        return output

    def _instantiate(self, __forced_tvars, name, *args, **kwargs):
        return self._instantiate_func(__forced_tvars, name, args=args, kwargs=kwargs)

    def __call__(cls, *args, **kwargs):
        return cls._instantiate(None, None, *args, **kwargs)

    def using(cls, name=None, **__forced_tvars):
        return lambda *args, **kwargs: cls._instantiate(__forced_tvars, name, *args, **kwargs)

    @property
    def __signature__(cls):
        # Implement so that `help` works properly on adapter definitions.
        parameters = [
            inspect.Parameter(
                input_def.name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=input_def.typ,
                default=cls._signature.defaults.get(input_def.name, inspect.Parameter.empty),
            )
            for input_def in cls._signature.inputs
        ]
        return inspect.Signature(parameters)


# Every AdapterDef instance represents an instance of a wiring-time input or output adapter
class AdapterDef:
    def __init__(self, inputs, scalars, tvars, adapterimpl):
        self._managerDef = None
        if self._managed:
            self._managerDef = scalars[0]
            scalars = scalars[1:]

        self._inputs = inputs
        self._scalars = scalars
        self._impl = adapterimpl
        self._tvars = tvars

    @property
    def _signature(self):
        return type(self)._signature

    def ts_inputs(self):
        return (((ts_idx, -1), input) for ts_idx, input in enumerate(self._inputs))


class InputAdapterDef(AdapterDef):
    def __init__(self, inputs, scalars, tvars, adapterimpl):
        assert len(inputs) == 0
        self._push_mode = scalars[-1]
        scalars = scalars[:-1]
        super().__init__(inputs, scalars, tvars, adapterimpl)
        self._output_def = None

    def _create(self, engine, memo):
        assert self._output_def is not None
        manager = None
        if self._managerDef:
            manager = memo.get(self._managerDef)
            if manager is None:
                manager = memo[self._managerDef] = self._managerDef._create(engine, memo)

        return self._impl(
            manager,
            engine,
            ContainerTypeNormalizer.normalized_type_to_actual_python_type(self._output_def.typ.typ),
            self._push_mode,
            tuple(ContainerTypeNormalizer.normalized_type_to_actual_python_type(s) for s in self._scalars),
        )

    def set_output_def(self, output_def):
        self._output_def = output_def


class OutputAdapterDef(AdapterDef):
    def __init__(self, inputs, scalars, tvars, adapterimpl):
        assert len(inputs) == 1
        super().__init__(inputs, scalars, tvars, adapterimpl)

    def _create(self, engine, memo):
        manager = None
        if self._managerDef:
            manager = memo.get(self._managerDef)
            if manager is None:
                manager = memo[self._managerDef] = self._managerDef._create(engine, memo)
        return self._impl(manager, engine, self._scalars)


def _adapterdef(BaseDef, name, create_method, out_type, manager_type, memoize=True, force_memoize=False, **kwargs):
    """

    :param BaseDef:
    :param name:
    :param create_method:
    :param out_type:
    :param manager_type:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).    :param kwargs:
    :return:
    """
    inputs = []
    defaults = {}
    is_input = out_type is not None

    if manager_type:
        inputs.append(InputDef("manager", manager_type, ArgKind.SCALAR, None, -1, 0))

    if out_type and not tstype.isTsType(out_type):
        raise TypeError(f"adapterdef out_type must be a ts[] type, got {out_type}")

    for arg, typ in kwargs.items():
        if isinstance(typ, tuple):
            typ, default = typ
            defaults[arg] = default
        elif defaults:
            raise SyntaxError("non-default argument follows default argument")

        arg_kind = ArgKind.TS if tstype.isTsType(typ) else ArgKind.SCALAR

        if arg_kind == ArgKind.TS and out_type is not None:
            raise ValueError("input adapters cannot have ts inputs")

        inputs.append(InputDef(arg, typ, arg_kind, None, -1, len(inputs)))

    if is_input:
        inputs.append(InputDef("push_mode", PushMode, ArgKind.SCALAR, None, -1, len(inputs)))
        defaults["push_mode"] = PushMode.NON_COLLAPSING

        outputs = (OutputDef(name=None, typ=out_type, kind=ArgKind.TS, ts_idx=0, shape=None),)
    else:
        outputs = tuple()

    signature = Signature(name, tuple(inputs), outputs, defaults)
    adaptertype = AdapterDefMeta(
        name,
        (BaseDef,),
        {
            "_signature": signature,
            "_managed": manager_type is not None,
            "memoize": memoize,
            "force_memoize": force_memoize,
            "_impl": create_method,
        },
    )
    return adaptertype


def input_adapter_def(name, adapterimpl, out_type, manager_type=None, memoize=True, force_memoize=False, **kwargs):
    """Create a graph representation of an input adapter defined in C++
    :param name: string name for the adapter
    :param adapterimpl: a C creator method exposed to python to create the c++ adapter impl
    :param out_type: the type of the output, should be a ts[] type.  Note this can use tvar types if a subsequent argument defines the tvar
    :param manager_type:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param kwargs: **kwargs will be passed through as arguments to the PullInputAdapter implementation
    :return:
    """

    def impl(mgr, engine, pytype, push_mode, scalars):
        return adapterimpl(mgr, engine, pytype, push_mode, scalars)

    return _adapterdef(
        InputAdapterDef, name, impl, out_type, manager_type, memoize=memoize, force_memoize=force_memoize, **kwargs
    )


status_adapter_def = input_adapter_def("status_adapter", _cspimpl._status_adapter, ts["T"], object, typ="T")


def py_pull_adapter_def(name, adapterimpl, out_type, memoize=True, force_memoize=False, **kwargs):
    """
    Create a graph representation of a python pull adapter.
    :param name:         string name for the adapter
    :param adapterimpl: a derived implementation of csp.impl.pulladapter.PullInputAdapter
    :param out_type:     the type of the output, should be a ts[] type.  Note this can use tvar types if a subsequent argument defines the tvar
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param kwargs:       **kwargs will be passed through as arguments to the PullInputAdapter implementation
    """

    def impl(mgr, engine, pytype, push_mode, scalars):
        return _cspimpl._pulladapter(mgr, engine, pytype, push_mode, (adapterimpl, scalars))

    return _adapterdef(
        InputAdapterDef, name, impl, out_type, manager_type=None, memoize=memoize, force_memoize=force_memoize, **kwargs
    )


def py_managed_adapter_def(name, adapterimpl, out_type, manager_type, memoize=True, force_memoize=False, **kwargs):
    """
    Create a graph representation of a python managed sim input adapter.
    :param name:         string name for the adapter
    :param adapterimpl: a derived implementation of csp.impl.adaptermanager.ManagedSimInputAdapter
    :param out_type:     the type of the output, should be a ts[] type.  Note this can use tvar types if a subsequent argument defines the tvar
    :param manager_type: the type of the graph time representation of the AdapterManager that will manage this adapter
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
           this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
           set of parameters).
    :param kwargs:       **kwargs will be passed through as arguments to the ManagedSimInputAdapter implementation
                         the first argument to the implementation will be the adapter manager impl instance
    """

    def impl(mgr, engine, pytype, push_mode, scalars):
        return _cspimpl._managedsimadapter(mgr, engine, pytype, push_mode, (adapterimpl, (mgr, *scalars)))

    return _adapterdef(
        InputAdapterDef, name, impl, out_type, manager_type, memoize=memoize, force_memoize=force_memoize, **kwargs
    )


def py_push_adapter_def(name, adapterimpl, out_type, manager_type=None, memoize=True, force_memoize=False, **kwargs):
    """
    :param name:
    :param adapterimpl:
    :param out_type:
    :param manager_type:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param kwargs:
    :return:
    """

    def impl(mgr, engine, pytype, push_mode, scalars):
        push_group = scalars[-1]
        scalars = scalars[:-1]
        if mgr is not None:
            scalars = (mgr,) + scalars
        return _cspimpl._pushadapter(mgr, engine, pytype, push_mode, (adapterimpl, push_group, scalars))

    return _adapterdef(
        InputAdapterDef,
        name,
        impl,
        out_type,
        manager_type,
        memoize=memoize,
        force_memoize=force_memoize,
        **kwargs,
        push_group=(object, None),
    )


def py_pushpull_adapter_def(
    name, adapterimpl, out_type, manager_type=None, memoize=True, force_memoize=False, **kwargs
):
    """
    :param name:
    :param adapterimpl:
    :param out_type:
    :param manager_type:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param kwargs:
    :return:
    """

    def impl(mgr, engine, pytype, push_mode, scalars):
        push_group = scalars[-1]
        scalars = scalars[:-1]
        if mgr is not None:
            scalars = (mgr,) + scalars
        return _cspimpl._pushpulladapter(mgr, engine, pytype, push_mode, (adapterimpl, push_group, scalars))

    return _adapterdef(
        InputAdapterDef,
        name,
        impl,
        out_type,
        manager_type,
        memoize=memoize,
        force_memoize=force_memoize,
        **kwargs,
        push_group=(object, None),
    )


# output adapters
def output_adapter_def(name, adapterimpl, manager_type=None, memoize=True, force_memoize=False, **kwargs):
    """

    :param name:
    :param adapterimpl:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param kwargs:
    :return:
    """
    return _adapterdef(
        OutputAdapterDef, name, adapterimpl, None, manager_type, memoize=memoize, force_memoize=force_memoize, **kwargs
    )


def py_output_adapter_def(name, adapterimpl, manager_type=None, memoize=True, force_memoize=False, **kwargs):
    """

    :param name:
    :param adapterimpl:
    :param memoize: Specify whether the node should be memoized (default True)
    :param force_memoize: If True, the node will be memoized even if csp.memoize(False) was called. Usually it should not be set, set
    this to True ONLY if memoization required to guarantee correctness of the function (i.e the function must be called at most once with the for each
    set of parameters).
    :param kwargs:
    :return:
    """

    def impl(mgr, engine, scalars):
        if mgr is not None:
            scalars = (mgr,) + scalars
        return _cspimpl._outputadapter(mgr, engine, (adapterimpl, scalars))

    return _adapterdef(
        OutputAdapterDef, name, impl, None, manager_type, memoize=memoize, force_memoize=force_memoize, **kwargs
    )


@override
def add_graph_output(
    key: object,
    input: tstype.ts["T"],  # noqa: F821
    tick_count: int = -1,
    tick_history: timedelta = timedelta(),
):
    # Stub for IDE auto-complete/static type checking
    ...


add_graph_output = output_adapter_def(  # noqa: F811
    "add_graph_output",
    _cspimpl._graph_output_adapter,
    key=object,
    input=tstype.ts["T"],
    tick_count=(int, -1),
    tick_history=(timedelta, timedelta()),
)
_graph_return_adapter = output_adapter_def(
    "_graph_return", _cspimpl._graph_return_adapter, key=object, input=tstype.ts["T"]
)
