"""Host registration for CSP-owned native struct views."""

import ast

from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.binding import StatePlan
from numba_cfunc_compiler.types.factory import HostTypeFactory

from csp.impl.struct import Struct as CspStruct
from csp.impl.wiring.csp_numba.struct_native import csp_struct_payload, register_symbols


class CspStructHostType(HostType):
    @classmethod
    def is_type_supported(cls, var_type):
        return isinstance(var_type, type) and issubclass(var_type, CspStruct) and var_type is not CspStruct

    def get_state_payload(self):
        return csp_struct_payload(self.value)

    def is_opaque_pointer(self):
        return True

    def state_plan(self, variable):
        plan = super().state_plan(variable)
        return StatePlan(
            before=plan.before,
            metadata={
                "struct_state_indices": (variable.array_idx,),
                "struct_state_sizes": (self.value._metadata_info()["size"],),
            },
        )

    @classmethod
    def try_parse_state(cls, node, var_name, globalns):
        try:
            struct_type = eval(ast.unparse(node.annotation.slice), globalns)
        except (NameError, AttributeError):
            return None
        if not cls.is_type_supported(struct_type):
            return None
        csp_struct_payload(struct_type)
        value = (
            struct_type()
            if node.value is None or isinstance(node.value, ast.Constant) and node.value.value is None
            else eval(ast.unparse(node.value), globalns)
        )
        if not isinstance(value, struct_type):
            raise TypeError(f"State '{var_name}' requires a {struct_type.__name__} initializer")
        return value, struct_type


def register():
    register_symbols()
    HostTypeFactory.register(CspStructHostType, priority=0)
