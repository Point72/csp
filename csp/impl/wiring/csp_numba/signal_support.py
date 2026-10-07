"""CSP callback inputs. Basket iterators yield slot positions."""

from numba import types
from numba_cfunc_compiler.core.variable_access import KeyedInputAccess, SourceAccess
from numba_cfunc_compiler.extension.callback_components import (
    CallbackComponent,
    CfuncParam,
    ComponentRegistry,
    MaterializationPhase,
)
from numba_cfunc_compiler.extension.input_sources import (
    SLOT_INDEX,
    KeyFilter,
    SourceField,
    register_input_source,
    register_keyed_input,
)
from numba_cfunc_compiler.types.factory import HostTypeFactory

from csp.impl.wiring.csp_numba.input_handlers import CspInputCategory

ACTIVE_SOURCE = register_input_source(
    identity=("csp", "signal.active.v1"),
    fields=(
        SourceField("valid", types.CPointer(types.int8)),
        SourceField("ticked", types.CPointer(types.int8)),
        SourceField("index", types.intp),
    ),
)


@ACTIVE_SOURCE.method("valid")
def signal_valid(signal):
    return signal.source.valid[signal.source.index] != 0


@ACTIVE_SOURCE.method("ticked")
def signal_ticked(signal):
    return signal.source.ticked[signal.source.index] != 0


ACTIVE_BASKET = register_keyed_input(
    identity=("csp", "basket.active.v1"),
    element_source=ACTIVE_SOURCE,
    fields=(
        SourceField("slots", types.CPointer(types.voidptr)),
        SourceField("valid_flags", types.CPointer(types.int8)),
        SourceField("ticked_flags", types.CPointer(types.int8)),
        SourceField("start", types.intp),
        SourceField("length", types.intp),
    ),
    element_projection={"valid": "valid_flags", "ticked": "ticked_flags", "index": SLOT_INDEX},
)
ACTIVE_BASKET.iter_method("keys", filter=KeyFilter.all())
ACTIVE_BASKET.iter_method("validkeys", filter=KeyFilter.nonzero("valid_flags"))
ACTIVE_BASKET.iter_method("tickedkeys", filter=KeyFilter.nonzero("ticked_flags"))


@ACTIVE_BASKET.method("valid")
def basket_valid(basket):
    for position in range(basket.length):
        if basket.valid_flags[basket.start + position] == 0:
            return False
    return True


@ACTIVE_BASKET.method("ticked")
def basket_ticked(basket):
    for position in range(basket.length):
        if basket.ticked_flags[basket.start + position] != 0:
            return True
    return False


class SignalComponent(CallbackComponent):
    id = "csp.signal"
    order = 0
    materialization_phase = MaterializationPhase.EXECUTE

    @property
    def cfunc_params(self):
        return [
            CfuncParam("inputs", "CPointer(voidptr)"),
            CfuncParam("input_ticked", "CPointer(int8)"),
            CfuncParam("input_valid", "CPointer(int8)"),
        ]

    def create_variables(self, inputs, factory):
        ordered = []
        for name, edge in inputs.input_analysis.get_by_category(CspInputCategory.SIGNAL).items():
            index = len(ordered)
            factory.add_variable(
                SourceAccess(
                    index,
                    HostTypeFactory.resolve(inputs.extract_python_type_fn(edge)),
                    name,
                    "inputs",
                    ACTIVE_SOURCE,
                    {"valid": "input_valid", "ticked": "input_ticked", "index": index},
                ),
                component=self.id,
            )
            ordered.append(edge)
        for name, basket in inputs.input_analysis.get_by_category(CspInputCategory.SIGNAL_SET).items():
            children = {}
            for key, edge in basket.items():
                index = len(ordered)
                children[key] = (index, HostTypeFactory.resolve(inputs.extract_python_type_fn(edge)))
                ordered.append(edge)
            factory.add_variable(
                KeyedInputAccess(
                    name,
                    children,
                    ACTIVE_BASKET,
                    {"slots": "inputs", "valid_flags": "input_valid", "ticked_flags": "input_ticked"},
                ),
                component=self.id,
            )
        return {"ordered_input_signals": tuple(ordered)}


def register():
    ComponentRegistry.register(SignalComponent())
