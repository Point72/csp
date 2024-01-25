from types import MethodType

from csp.impl.__cspimpl import _cspimpl
from csp.impl.types.common_definitions import ArgKind, OutputDef
from csp.impl.types.container_type_normalizer import ContainerTypeNormalizer
from csp.impl.wiring import runtime, tstype
from csp.impl.wiring.edge import Edge
from csp.impl.wiring.node import NodeDef
from csp.impl.wiring.signature import Signature


class DynamicNodeDef:
    def __init__(self, name, inputs, dynamic_args, sub_graph, *raw_args):
        self._name = name
        self._inputs = inputs
        self._outputs = None

        self._sub_graph = sub_graph
        self._raw_args = list(raw_args)
        self._dynamic_args = tuple(dynamic_args)

    def ts_inputs(self):
        """return generator of proper (( ts_idx, basket_idx) ,edge) inputs adjusted for alarms"""
        ts_idx = 0
        for input in self._inputs:
            # TODO allow basket inputs to dynamic
            # if isinstance(input, list):
            #    for basket_idx, basket_input in enumerate(input):
            #        yield ((ts_idx, basket_idx), basket_input)
            # elif isinstance(input, dict):
            #    for basket_idx, basket_input in enumerate(input.values()):
            #        yield ((ts_idx, basket_idx), basket_input)
            # else:
            yield ((ts_idx, -1), input)
            ts_idx += 1

    def _create(self, engine, memo):
        # Avoid capturing self
        sub_graph = self._sub_graph

        def builder(sub_engine, adjusted_args):
            # TODO figure out the rest of the build_graph args...
            # TODO figure out how to properly use existing memo... we want some things to get cached, but not others...
            # g = runtime.build_graph(sub_graph, *args, starttime=starttime, endtime=endtime, config=config, **kwargs)

            # Note that we do NOT copy the memo over from the outter graph.  We may want to revisit this but it would only
            # make sense if we enable memoization across outter / inner graphs ( need to pass csp_memoization instance through )
            # There is an unsolvable issue though, I think.  if we memoize something as trivial a csp.const(1) it will never tick
            # in the dynamic
            # However, We do want to copy over adapter managers and external inputs, including our trigger
            m = {inp.nodedef: memo.get(inp.nodedef) for inp in self._inputs}
            g = runtime.build_graph(sub_graph, *adjusted_args)
            return runtime._build_engine(sub_engine, g, m)

        inputs = NodeDef._normalize_inputs_to_cpp_types(self._inputs, self._name)
        # output name is None for single nameless output, convert to ''
        outputs = tuple(o.name if o.name is not None else "" for o in self._outputs)
        node = _cspimpl.PyDynamicNode(engine, self._name, inputs, outputs, builder, self._dynamic_args, self._raw_args)

        for idx, output in enumerate(self._outputs):
            node.create_output(
                idx, (None, ContainerTypeNormalizer.normalized_type_to_actual_python_type(output.typ.__args__[1].typ))
            )

        return node


def _set_dynbasket_args(args, trigger, depth=0):
    """helper method to set trigger input on snapkey and attach args"""
    for arg in args:
        if isinstance(arg, (tstype.SnapKeyType, tstype.AttachType)):
            if depth > 0:
                raise TypeError("csp.snap and csp.attach are not supported as members of containers")
            arg._dyn_basket = trigger
        elif isinstance(arg, (tuple, list)):
            _set_dynbasket_args(arg, trigger, depth + 1)
        elif isinstance(arg, dict):
            _set_dynbasket_args(arg.values(), trigger, depth + 1)


def dynamic(trigger, sub_graph, *args, **kwargs):
    """define a dynamic sub-graph
    :param trigger:   dynamic basket trigger used to trigger creation / destruction of dynamic sub-graphs
    :param sub_graph: csp.graph method to be used to create dynamic sub-graphs
    remaining args/kwargs are passed to sub_graph upon dynamic instantiation
    """

    trigger_keytype = trigger.tstype.__args__[0].typ
    sig_outputs = []
    for out in sub_graph._signature._outputs:
        if out.kind.is_basket():
            raise TypeError("csp.dynamic does not support basket outputs of sub_graph")
        sig_outputs.append(
            OutputDef(
                out.name,
                tstype.DynamicBasket[trigger_keytype, out.typ.typ],
                ArgKind.DYNAMIC_BASKET_TS,
                out.ts_idx,
                None,
            )
        )

    # update SnapKey and Attach args with the input basket
    _set_dynbasket_args(args, trigger)

    # if `sub_graph` is a bound method (e.g. calling csp.dynamic(trigger, self.sub_graph),
    # inject `self` from the bound method into the arguments.
    # NOTE: this is only done for the parse_inputs and raw_args calls, we will
    # undo this a few lines below
    if isinstance(sub_graph, MethodType):
        adjusted_args = (sub_graph.__self__,) + args
    else:
        adjusted_args = args

    # This is just for extracting tvars and type checking at csp.dynamic wiring time
    inputs, scalars, tvars = sub_graph._signature.parse_inputs(None, *adjusted_args, **kwargs)

    inputs = [trigger] + list(inputs)
    raw_args = sub_graph._signature.flatten_args(*adjusted_args, **kwargs)

    # undo the work from above if necessary
    if args is not adjusted_args:
        # purge `self`, required to be first arg
        raw_args = raw_args[1:]

    # Collect tuples of ( ts index, scalar args index, dynamic arg type ) for dynamic argument processing
    dynamic_args = []
    ts_idx = len(inputs)
    for idx, arg in enumerate(raw_args):
        if isinstance(arg, tstype.SnapType):
            dynamic_args.append((ts_idx, idx, "snap"))
            raw_args[idx] = None
            inputs.append(arg.edge)
            ts_idx += 1
        elif isinstance(arg, tstype.SnapKeyType):
            dynamic_args.append((0, idx, "snapkey"))
            raw_args[idx] = None
        elif isinstance(arg, tstype.AttachType):
            # For now we dont allow csp.attach on anything other than trigger, could be revisited
            # Allowing arbitrary baskets opens up issues with the key disappearing from the basket.
            dynamic_args.append((0, idx, "attach"))

            # For csp.attach, we prep the edge as the proper single ts nodedef output.  dynamic key will set
            # the proper edgeid when the time comes
            raw_args[idx] = Edge(arg.value_tstype, arg.dyn_basket.nodedef, arg.dyn_basket.output_idx)

    node_name = f"dynamic<{sub_graph._signature._name}>"
    nodedef = DynamicNodeDef(node_name, inputs, dynamic_args, sub_graph, *raw_args)

    # create a stub-signature to generate our dynamic basket outputs
    signature = Signature(
        sub_graph._signature._name, sub_graph._signature._inputs, sig_outputs, sub_graph._signature._defaults
    )
    outputs, output_types = signature.create_outputs(nodedef, tvars)
    nodedef._outputs = output_types
    return outputs
