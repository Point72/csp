import ast

from numba_type_utils.utils.ast import AST
from numba_type_utils.method_factory import MethodBase
from numba_type_utils.variable_factory import VoidPtrSource
from numba_type_utils.source_registry import CfuncParam, SourceCategory, SourceInitFilter

__all__ = [
    "INPUTS_ARRAY_NAME",
    "TICKED_ARRAY_NAME",
    "VALID_ARRAY_NAME",
    "Ticked",
    "Valid",
    "SignalSource",
    "SignalCategory",
]


INPUTS_ARRAY_NAME = "inputs"
TICKED_ARRAY_NAME = "input_ticked"
VALID_ARRAY_NAME = "input_valid"


def _build_signal_flag_check(var, array_name: str) -> ast.Compare:
    from csp.impl.wiring.csp_numba.signal_set_support import DynamicSignalAccess

    if isinstance(var, DynamicSignalAccess):
        return ast.Compare(
            left=ast.Subscript(
                value=ast.Name(id=array_name, ctx=ast.Load()),
                slice=var._get_effective_index(),
                ctx=ast.Load(),
            ),
            ops=[ast.NotEq()],
            comparators=[ast.Constant(value=0)],
        )

    if isinstance(var, SignalSource):
        return ast.Compare(
            left=AST.array_access(array_name, var.array_idx),
            ops=[ast.NotEq()],
            comparators=[ast.Constant(value=0)],
        )

    raise TypeError(f"called signal status check on unsupported source {type(var).__name__}")


class Ticked(MethodBase):
    @classmethod
    def get_name(cls) -> str:
        return "ticked"

    @staticmethod
    def handle(var, args):
        """Create: ticked_array[position] != 0"""
        if isinstance(var, SignalSource):
            ticked_array = var.get_ticked_array_name()
            if ticked_array is None:
                raise TypeError(f"called ticked on {var.name} which has no ticked array configured")
            return _build_signal_flag_check(var, ticked_array)

        return _build_signal_flag_check(var, TICKED_ARRAY_NAME)


class Valid(MethodBase):
    @classmethod
    def get_name(cls) -> str:
        return "valid"

    @staticmethod
    def handle(var, args):
        """Create: valid_array[position] != 0 using the configured array."""
        if isinstance(var, SignalSource):
            valid_array = var.get_valid_array_name()
            if valid_array is None:
                raise TypeError(f"called valid on {var.name} which has no valid array configured")
            return _build_signal_flag_check(var, valid_array)

        return _build_signal_flag_check(var, VALID_ARRAY_NAME)


class SignalSource(VoidPtrSource):
    """VoidPtrSource with ticked/valid array support for signals."""

    def __init__(
        self,
        array_idx: int,
        type,
        name: str,
        storage_location: str,
        valid_array_name: str,
        ticked_array_name: str = None,
        supported_methods=None,
        force_opaque: bool = False,
        skip_pre_read: bool = False,
    ):
        self._valid_array_name = valid_array_name
        self._ticked_array_name = ticked_array_name
        super().__init__(
            array_idx=array_idx,
            type=type,
            name=name,
            storage_location=storage_location,
            supported_methods=supported_methods,
            force_opaque=force_opaque,
            skip_pre_read=skip_pre_read,
        )

    def get_valid_array_name(self):
        return self._valid_array_name

    def get_ticked_array_name(self):
        return self._ticked_array_name


class SignalCategory(SourceCategory):
    """Active signal inputs (inputs, input_ticked, input_valid)."""

    id = "csp.signal"
    order = 0
    init_filter = SourceInitFilter.ON_EXECUTE

    @property
    def cfunc_params(self):
        return [
            CfuncParam("inputs", "CPointer(voidptr)"),
            CfuncParam("input_ticked", "CPointer(int8)"),
            CfuncParam("input_valid", "CPointer(int8)"),
        ]

    def _create_signal_source(
        self, array_idx: int, var_type, name: str, *, skip_pre_read: bool = False
    ) -> SignalSource:
        return SignalSource(
            array_idx=array_idx,
            type=var_type,
            name=name,
            storage_location=INPUTS_ARRAY_NAME,
            valid_array_name=VALID_ARRAY_NAME,
            ticked_array_name=TICKED_ARRAY_NAME,
            force_opaque=True,
            skip_pre_read=skip_pre_read,
            supported_methods=[Valid, Ticked],
        )

    def create_variables(self, info, factory):
        from numba_type_utils.type_factory import TypeFactory
        from csp.impl.wiring.csp_numba.signal_set_support import SignalSetSource

        input_idx = 0
        info.ordered_input_signals = []

        for name, signal_obj in info.input_analysis.get_by_category("signal").items():
            var_type = TypeFactory.get_type(info.extract_python_type_fn(signal_obj))
            var = self._create_signal_source(input_idx, var_type, name)
            factory.add_variable(var, category=self.id)
            info.ordered_input_signals.append(signal_obj)
            input_idx += 1

        for set_name, signal_set in info.input_analysis.get_by_category("signal_set").items():
            key_to_child_name = {}
            start_idx = input_idx
            element_type = None
            for key, signal_obj in signal_set.items():
                child_name = f"{set_name}__{key}"
                key_to_child_name[key] = child_name
                child_type = TypeFactory.get_type(info.extract_python_type_fn(signal_obj))
                if element_type is None:
                    element_type = child_type
                child_var = self._create_signal_source(input_idx, child_type, child_name, skip_pre_read=True)
                factory.add_variable(child_var, category=self.id)
                info.ordered_input_signals.append(signal_obj)
                input_idx += 1
            factory.add_variable(
                SignalSetSource(
                    name=set_name,
                    key_to_child_name=key_to_child_name,
                    start_idx=start_idx,
                    element_type=element_type,
                ),
                category=self.id,
            )

    def get_result_metadata(self, info):
        return {"ordered_input_signals": list(info.ordered_input_signals)}
