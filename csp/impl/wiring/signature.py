import itertools

from csp.impl.constants import UNSET
from csp.impl.types import tstype
from csp.impl.types.common_definitions import ArgKind, InputDef, OutputBasketContainer, OutputDef
from csp.impl.types.generic_values_resolver import GenericValuesResolver
from csp.impl.types.instantiation_type_resolver import InputInstanceTypeResolver
from csp.impl.types.tstype import ts
from csp.impl.wiring.context import Context
from csp.impl.wiring.edge import Edge
from csp.impl.wiring.outputs import OutputsContainer
from csp.impl.wiring.special_output_names import UNNAMED_OUTPUT_NAME


class Signature:
    def __init__(self, name, inputs, outputs, defaults, special_outputs=None):
        self._name = name
        self._inputs = inputs
        self._outputs = outputs
        # Special outputs are all the special outputs of the node/graph
        # The don't have to be included in the outputs.
        # Currently for node, the special outputs are included in the outputs
        # while for graph they are not.
        # Conceptually for now we try to preserve the following semantics:
        # 1. OutputsContainer contains all outputs that should be available to user of the node/graph
        # 2. Special outputs includes all "special" outputs whether they are visible to user or not.
        self._special_outputs = special_outputs
        self._defaults = defaults

        self._input_map = {arg[0]: arg for idx, arg in enumerate(inputs)}
        self._output_map = {arg[0]: arg for idx, arg in enumerate(outputs)}
        self._special_output_map = {arg[0]: arg for arg in self._special_outputs} if self._special_outputs else {}
        self._ts_inputs = [x for x in self._inputs if x.kind.is_any_ts()]
        self._alarms = [x for x in self._inputs if x.kind.is_alarm()]
        self._scalars = [x for x in self._inputs if x.kind.is_scalar()]
        self._num_alarms = len(self._alarms)

    def copy(self, drop_alarms=False):
        if drop_alarms:
            new_inputs = []
            cur_ts_dx = 0
            for input in self._inputs:
                if input.kind == ArgKind.ALARM:
                    continue
                else:
                    new_inputs.append(
                        InputDef(
                            name=input.name,
                            typ=input.typ,
                            kind=input.kind,
                            basket_kind=input.basket_kind,
                            ts_idx=cur_ts_dx,
                            arg_idx=input.arg_idx,
                        )
                    )
                    cur_ts_dx += 1
        else:
            new_inputs = self._inputs
        return Signature(
            self._name, new_inputs, self._outputs, defaults=self._defaults, special_outputs=self._special_outputs
        )

    def flatten_args(self, *args, **kwargs):
        if self._defaults:
            new_kwargs = self._defaults.copy()
            # Remove defaults that have been passed as positional
            for idx in range(len(args)):
                new_kwargs.pop(self._inputs[self._num_alarms + idx].name, None)
            new_kwargs.update(kwargs)
            kwargs = new_kwargs

        # Flatten out kwargs
        flat_args = list(args) + [UNSET] * len(kwargs)
        if len(flat_args) != self.num_args():
            raise TypeError("%s takes %d arguments but given %d" % (self._name, self.num_args(), len(flat_args)))

        for k, value in kwargs.items():
            input = self._input_map.get(k, None)
            if input is None:
                raise TypeError("%s got an unexpected keyword argument '%s'" % (self._name, k))
            arg = input
            if arg.arg_idx < len(args):
                raise TypeError('%s got multiple value for argument "%s"' % (self._name, k))

            flat_args[arg.arg_idx] = value

        return flat_args

    def parse_inputs(self, forced_tvars, *args, allow_subtypes=True, allow_none_ts=False, **kwargs):
        from csp.utils.object_factory_registry import Injected

        flat_args = self.flatten_args(*args, **kwargs)

        for i, arg in enumerate(flat_args):
            if isinstance(arg, Injected):
                flat_args[i] = arg.value

        type_resolver = InputInstanceTypeResolver(
            function_name=self._name,
            input_definitions=self._inputs[self._num_alarms :],
            arguments=flat_args,
            forced_tvars=forced_tvars,
            allow_none_ts=allow_none_ts,
        )

        # We need to do some special handling here of int arguments connected to float
        tvars = type_resolver.tvars
        resolved_ts_inputs = type_resolver.ts_inputs
        non_alarm_inputs = (e for e in self._ts_inputs if e.kind != ArgKind.ALARM)
        for i, (input, resolved_input) in enumerate(zip(non_alarm_inputs, resolved_ts_inputs)):
            if isinstance(resolved_input, Edge):
                if getattr(resolved_input.tstype, "typ", None) is int:
                    declared_type = GenericValuesResolver.resolve_generic_values(input.typ, tvars)
                    if declared_type.typ is float:
                        from csp.baselib import cast_int_to_float

                        resolved_ts_inputs[i] = cast_int_to_float(resolved_input)
            elif isinstance(resolved_input, list):
                declared_type = GenericValuesResolver.resolve_generic_values(input.typ, tvars)
                for j, edge in enumerate(resolved_input):
                    if edge.tstype.typ is int:
                        if declared_type.__args__[0].typ is float:
                            from csp.baselib import cast_int_to_float

                            resolved_input[j] = cast_int_to_float(resolved_input[j])
            elif isinstance(resolved_input, dict):
                declared_type = GenericValuesResolver.resolve_generic_values(input.typ, tvars)
                for k, edge in resolved_input.items():
                    if edge.tstype.typ is int:
                        if declared_type.__args__[1].typ is float:
                            from csp.baselib import cast_int_to_float

                            resolved_input[k] = cast_int_to_float(resolved_input[k])

        return tuple(type_resolver.ts_inputs), tuple(type_resolver.scalar_inputs), type_resolver.tvars

    def _create_alarms(self, tvars):
        alarms = []
        for alarm in self._alarms:
            alarm = InputDef(
                name=alarm.name,
                typ=GenericValuesResolver.resolve_generic_values(alarm.typ, tvars),
                kind=alarm.kind,
                basket_kind=alarm.basket_kind,
                ts_idx=alarm.ts_idx,
                arg_idx=alarm.arg_idx,
            )
            alarms.append(alarm)
        return alarms

    def _create_output_edges(self, nodedef, output, output_idx: int):
        if output.kind == ArgKind.BASKET_TS:
            basket_idx = itertools.count()

            num_edges = output.shape if isinstance(output.shape, int) else len(output.shape)
            edges = [Edge(output.typ, nodedef, output_idx, basket_idx=next(basket_idx)) for i in range(num_edges)]

            if isinstance(output.shape, int):
                return edges
            else:
                return dict(zip(output.shape, edges))
        else:
            return Edge(output.typ, nodedef, output_idx)

    def _resolve_basket_types_and_normalize(self, output_type, basket_shape_eval_inputs):
        if isinstance(output_type, OutputBasketContainer):
            assert basket_shape_eval_inputs is not None
            typ, shape = output_type.get_type_and_shape(*basket_shape_eval_inputs)
            return typ, shape
        elif tstype.isTsDynamicBasket(output_type):
            return output_type, None
        elif tstype.isTsType(output_type):
            return output_type.typ, None
        else:
            # Graphs can have outputs but with no shape
            assert tstype.isTsBasket(output_type)
            return output_type, None

    @property
    def special_outputs(self):
        return self._special_outputs

    def _iter_all_outputs(self):
        yield from self._outputs
        if self._special_outputs:
            for o in self._special_outputs:
                if o not in self._outputs:
                    yield o

    def resolve_output_definitions(self, basket_shape_eval_inputs, tvars):
        output_definitions = []
        for output_def in self._iter_all_outputs():
            resolved_typ = GenericValuesResolver.resolve_generic_values(output_def.typ, tvars)
            resolved_typ, shape = self._resolve_basket_types_and_normalize(resolved_typ, basket_shape_eval_inputs)
            if resolved_typ is not output_def.typ or shape is not None:
                # dynamic baskets we dont want the type wrapped in ts[]
                tstyp = ts[resolved_typ] if output_def.kind != ArgKind.DYNAMIC_BASKET_TS else resolved_typ
                output_def = OutputDef(
                    name=output_def.name, typ=tstyp, kind=output_def.kind, ts_idx=output_def.ts_idx, shape=shape
                )
            output_definitions.append(output_def)
        return tuple(output_definitions)

    def resolve_basket_key_type(self, name_or_idx, tvars):
        output = self.output(name_or_idx)
        return GenericValuesResolver.resolve_generic_values(output.typ.typ.__args__[0], tvars)

    def create_outputs(self, nodedef, tvars, basket_shape_eval_inputs=None):
        """returns output edges to be consumed by user in graph wiring, as well as flat output type list
        for engine building"""
        if not self._outputs:
            if not hasattr(Context.TLS, "instance"):
                raise RuntimeError("graph must be created under a wiring context")
            Context.TLS.instance.roots.append(nodedef)
            return None, ()

        output_definitions = self.resolve_output_definitions(basket_shape_eval_inputs, tvars)

        if len(output_definitions) > 1 and output_definitions[0].name is None:
            output_definitions = (
                OutputDef(
                    name=UNNAMED_OUTPUT_NAME,
                    typ=output_definitions[0].typ,
                    kind=output_definitions[0].kind,
                    ts_idx=output_definitions[0].ts_idx,
                    shape=output_definitions[0].shape,
                ),
            ) + output_definitions[1:]

        if output_definitions[0].name is not None:
            idx = itertools.count()
            return OutputsContainer(
                **{o.name: self._create_output_edges(nodedef, o, next(idx)) for o in output_definitions}
            ), output_definitions

        outdef = output_definitions[0]
        return self._create_output_edges(nodedef, outdef, 0), (outdef,)

    def alarms(self):
        return self._alarms

    def raw_inputs(self):
        return self._inputs

    def input(self, name, allow_missing=False):
        if allow_missing:
            return self._input_map.get(name)
        else:
            return self._input_map[name]

    def ts_input_by_id(self, id):
        return self._ts_inputs[id]

    def output(self, name_or_idx, allow_missing=False):
        """returns output by name or by index.
        allow_missing=True will return None only for lookup by name"""
        if isinstance(name_or_idx, int):
            return self._outputs[name_or_idx]

        return self._output_map.get(name_or_idx) if allow_missing else self._output_map[name_or_idx]

    def num_args(self):
        return len(self._inputs) - self._num_alarms

    @property
    def inputs(self):
        return self._inputs

    @property
    def ts_inputs(self):
        return self._ts_inputs

    @property
    def scalars(self):
        return self._scalars
