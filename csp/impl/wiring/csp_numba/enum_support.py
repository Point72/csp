"""CSP IntEnum storage and literal binding for compiled nodes."""

import ast
import inspect
from dataclasses import dataclass
from enum import IntEnum

from numba_cfunc_compiler.core.analysis import ParameterInfo
from numba_cfunc_compiler.types.base import HostType
from numba_cfunc_compiler.types.binding import ConstantPlan
from numba_cfunc_compiler.types.builtin.enum.host import register_enum_family
from numba_cfunc_compiler.types.factory import HostTypeFactory
from numba_cfunc_compiler.utils.ast import AST


def register_family(enum_type):
    return register_enum_family(
        enum_type,
        abi_values={member.name: int(member.value) for member in enum_type},
        scalar_storage="int64",
        set_bits=None,
    )


@dataclass(frozen=True)
class CspEnumHostType(HostType):
    @classmethod
    def is_type_supported(cls, var_type):
        return isinstance(var_type, type) and issubclass(var_type, IntEnum)

    def get_numba_type_name(self):
        return "int64"

    def get_state_payload(self):
        return register_family(self.value).state_payload()

    def constant_plan(self, name, value, call_globals, slot_index):
        binding = register_family(self.value)
        literal_name = f"_csp_enum_constant_{name}"
        call_globals[literal_name] = binding.literal(value.name)
        expression = AST.function_call(literal_name)
        return ConstantPlan((AST.assignment(name, expression),), ast.Name(id=name, ctx=ast.Load()))

    @classmethod
    def try_parse_input(cls, param: inspect.Parameter, ann):
        if cls.is_type_supported(ann):
            return ParameterInfo(expected_type=ann)
        return None

    @classmethod
    def try_parse_state(cls, node, var_name, globalns):
        try:
            enum_type = eval(ast.unparse(node.annotation.slice), globalns)
        except (NameError, AttributeError):
            return None
        if not cls.is_type_supported(enum_type):
            return None
        value = eval(ast.unparse(node.value), globalns)
        if not isinstance(value, enum_type):
            raise TypeError(f"State '{var_name}' requires a {enum_type.__name__} initializer")
        register_family(enum_type)
        return value, enum_type


def register():
    HostTypeFactory.register(CspEnumHostType, priority=0)
