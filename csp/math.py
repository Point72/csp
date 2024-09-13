import math
from functools import lru_cache
from typing import List, TypeVar, get_origin

import numpy as np

import csp
from csp.impl.types.tstype import ts
from csp.impl.wiring import node
from csp.lib import _cspmathimpl
from csp.typing import Numpy1DArray, NumpyNDArray

__all__ = [
    "abs",
    "add",
    "and_",
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctanh",
    "bitwise_not",
    "cos",
    "cosh",
    "divide",
    "eq",
    "erf",
    "exp",
    "exp2",
    "floordiv",
    "ge",
    "gt",
    "le",
    "ln",
    "log10",
    "log2",
    "lt",
    "max",
    "min",
    "mod",
    "multiply",
    "ne",
    "neg",
    "not_",
    "or_",
    "pos",
    "pow",
    "sin",
    "sinh",
    "sqrt",
    "sub",
    "tan",
    "tanh",
]

T = TypeVar("T")
U = TypeVar("U")


@node(cppimpl=_cspmathimpl.bitwise_not)
def bitwise_not(x: ts[int]) -> ts[int]:
    return ~x


@node(cppimpl=_cspmathimpl.not_, name="not_")
def not_(x: ts[bool]) -> ts[bool]:
    """boolean not"""
    if csp.ticked(x):
        return not x


@node
def andnode(x: List[ts[bool]]) -> ts[bool]:
    if csp.valid(x):
        return all(x.validvalues())


def and_(*inputs):
    """binary and of basket of ts[ bool ]. Note that all inputs must be valid
    before any value is returned"""
    return andnode(list(inputs))


@node
def ornode(x: List[ts[bool]]) -> ts[bool]:
    if csp.valid(x):
        return any(x.validvalues())


def or_(*inputs):
    """binary or of ts[ bool ] inputs.  Note that all inputs must be valid
    before any value is returned"""
    return ornode(list(inputs))


# Math/comparison binary operators are supported in C++ only for (int,int) and
# (float, float) arguments. For all other types, the Python implementation is used.

MATH_OPS = [
    # binary
    "add",
    "sub",
    "multiply",
    "divide",
    "pow",
    "max",
    "min",
    "floordiv",
    "mod",
    # unary
    "pos",
    "neg",
    "abs",
    "ln",
    "log2",
    "log10",
    "exp",
    "exp2",
    "sqrt",
    "erf",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
]

COMP_OPS = ["eq", "ne", "lt", "gt", "le", "ge"]

MATH_COMP_OPS_CPP = {
    # binary math
    ("add", "float"): _cspmathimpl.add_f,
    ("add", "int"): _cspmathimpl.add_i,
    ("sub", "float"): _cspmathimpl.sub_f,
    ("sub", "int"): _cspmathimpl.sub_i,
    ("multiply", "float"): _cspmathimpl.mul_f,
    ("multiply", "int"): _cspmathimpl.mul_i,
    ("divide", "float"): _cspmathimpl.div_f,
    ("divide", "int"): _cspmathimpl.div_i,
    ("pow", "float"): _cspmathimpl.pow_f,
    ("pow", "int"): _cspmathimpl.pow_i,
    ("max", "float"): _cspmathimpl.max_f,
    ("max", "int"): _cspmathimpl.max_i,
    ("max", "np"): np.maximum,
    ("min", "float"): _cspmathimpl.min_f,
    ("min", "int"): _cspmathimpl.min_i,
    ("min", "np"): np.minimum,
    # unary math
    ("abs", "float"): _cspmathimpl.abs_f,
    ("abs", "int"): _cspmathimpl.abs_i,
    ("abs", "np"): np.abs,
    ("ln", "float"): _cspmathimpl.ln_f,
    ("ln", "int"): _cspmathimpl.ln_i,
    ("ln", "np"): np.log,
    ("log2", "float"): _cspmathimpl.log2_f,
    ("log2", "int"): _cspmathimpl.log2_i,
    ("log2", "np"): np.log2,
    ("log10", "float"): _cspmathimpl.log10_f,
    ("log10", "int"): _cspmathimpl.log10_i,
    ("log10", "np"): np.log10,
    ("exp", "float"): _cspmathimpl.exp_f,
    ("exp", "int"): _cspmathimpl.exp_i,
    ("exp", "np"): np.exp,
    ("exp2", "float"): _cspmathimpl.exp2_f,
    ("exp2", "int"): _cspmathimpl.exp2_i,
    ("exp2", "np"): np.exp2,
    ("sqrt", "float"): _cspmathimpl.sqrt_f,
    ("sqrt", "int"): _cspmathimpl.sqrt_i,
    ("sqrt", "np"): np.sqrt,
    ("erf", "float"): _cspmathimpl.erf_f,
    ("erf", "int"): _cspmathimpl.erf_i,
    # ("erf", "np"): np.erf,  # erf is in scipy, worth it to import?
    ("sin", "float"): _cspmathimpl.sin_f,
    ("sin", "int"): _cspmathimpl.sin_i,
    ("sin", "np"): np.sin,
    ("cos", "float"): _cspmathimpl.cos_f,
    ("cos", "int"): _cspmathimpl.cos_i,
    ("cos", "np"): np.cos,
    ("tan", "float"): _cspmathimpl.tan_f,
    ("tan", "int"): _cspmathimpl.tan_i,
    ("tan", "np"): np.tan,
    ("arcsin", "float"): _cspmathimpl.asin_f,
    ("arcsin", "int"): _cspmathimpl.asin_i,
    ("arcsin", "np"): np.arcsin,
    ("arccos", "float"): _cspmathimpl.acos_f,
    ("arccos", "int"): _cspmathimpl.acos_i,
    ("arccos", "np"): np.arccos,
    ("arctan", "float"): _cspmathimpl.atan_f,
    ("arctan", "int"): _cspmathimpl.atan_i,
    ("arctan", "np"): np.arctan,
    ("sinh", "float"): _cspmathimpl.sinh_f,
    ("sinh", "int"): _cspmathimpl.sinh_i,
    ("sinh", "np"): np.sinh,
    ("cosh", "float"): _cspmathimpl.cosh_f,
    ("cosh", "int"): _cspmathimpl.cosh_i,
    ("cosh", "np"): np.cosh,
    ("tanh", "float"): _cspmathimpl.tanh_f,
    ("tanh", "int"): _cspmathimpl.tanh_i,
    ("tanh", "np"): np.tanh,
    ("arcsinh", "float"): _cspmathimpl.asinh_f,
    ("arcsinh", "int"): _cspmathimpl.asinh_i,
    ("arcsinh", "np"): np.arcsinh,
    ("arccosh", "float"): _cspmathimpl.acosh_f,
    ("arccosh", "int"): _cspmathimpl.acosh_i,
    ("arccosh", "np"): np.arccosh,
    ("arctanh", "float"): _cspmathimpl.atanh_f,
    ("arctanh", "int"): _cspmathimpl.atanh_i,
    ("arctanh", "np"): np.arctanh,
    # binary comparator
    ("eq", "float"): _cspmathimpl.eq_f,
    ("eq", "int"): _cspmathimpl.eq_i,
    ("ne", "float"): _cspmathimpl.ne_f,
    ("ne", "int"): _cspmathimpl.ne_i,
    ("lt", "float"): _cspmathimpl.lt_f,
    ("lt", "int"): _cspmathimpl.lt_i,
    ("gt", "float"): _cspmathimpl.gt_f,
    ("gt", "int"): _cspmathimpl.gt_i,
    ("le", "float"): _cspmathimpl.le_f,
    ("le", "int"): _cspmathimpl.le_i,
    ("ge", "float"): _cspmathimpl.ge_f,
    ("ge", "int"): _cspmathimpl.ge_i,
}


@lru_cache(maxsize=512)
def define_binary_op(name, op_lambda):
    float_out_type, int_out_type, generic_out_type = [None] * 3
    if name in COMP_OPS:
        float_out_type = bool
        int_out_type = bool
        generic_out_type = bool
    elif name in MATH_OPS:
        float_out_type = float
        if name != "divide":
            int_out_type = int
            generic_out_type = "T"
        else:
            int_out_type = float
            generic_out_type = float

    from csp.impl.wiring.node import _node_internal_use

    @_node_internal_use(cppimpl=MATH_COMP_OPS_CPP.get((name, "float"), None), name=name)
    def float_type(x: ts[float], y: ts[float]) -> ts[float_out_type]:
        if csp.valid(x, y):
            return op_lambda(x, y)

    @_node_internal_use(cppimpl=MATH_COMP_OPS_CPP.get((name, "int"), None), name=name)
    def int_type(x: ts[int], y: ts[int]) -> ts[int_out_type]:
        if csp.valid(x, y):
            return op_lambda(x, y)

    numpy_func = MATH_COMP_OPS_CPP.get((name, "np"), op_lambda)

    @_node_internal_use(name=name)
    def numpy_type(x: ts["T"], y: ts["U"]) -> ts[np.ndarray]:
        if csp.valid(x, y):
            return numpy_func(x, y)

    @_node_internal_use(name=name)
    def generic_type(x: ts["T"], y: ts["T"]) -> ts[generic_out_type]:
        if csp.valid(x, y):
            return op_lambda(x, y)

    def comp(x: ts["T"], y: ts["U"]):
        if get_origin(x.tstype.typ) in [Numpy1DArray, NumpyNDArray] or get_origin(y.tstype.typ) in [
            Numpy1DArray,
            NumpyNDArray,
        ]:
            return numpy_type(x, y)
        elif x.tstype.typ is float and y.tstype.typ is float:
            return float_type(x, y)
        elif x.tstype.typ is int and y.tstype.typ is int:
            return int_type(x, y)
        return generic_type(x, y)

    comp.__name__ = name
    return comp


@lru_cache(maxsize=512)
def define_unary_op(name, op_lambda):
    float_out_type, int_out_type, generic_out_type = [None] * 3
    if name in COMP_OPS:
        float_out_type = bool
        int_out_type = bool
        generic_out_type = bool
    elif name in MATH_OPS:
        float_out_type = float
        if name in ("abs",):
            int_out_type = int
            generic_out_type = "T"
        else:
            int_out_type = float
            generic_out_type = float

    from csp.impl.wiring.node import _node_internal_use

    @_node_internal_use(cppimpl=MATH_COMP_OPS_CPP.get((name, "float"), None), name=name)
    def float_type(x: ts[float]) -> ts[float_out_type]:
        if csp.valid(x):
            return op_lambda(x)

    @_node_internal_use(cppimpl=MATH_COMP_OPS_CPP.get((name, "int"), None), name=name)
    def int_type(x: ts[int]) -> ts[int_out_type]:
        if csp.valid(x):
            return op_lambda(x)

    numpy_func = MATH_COMP_OPS_CPP.get((name, "np"), op_lambda)

    @_node_internal_use(name=name)
    def numpy_type(x: ts[np.ndarray]) -> ts[np.ndarray]:
        if csp.valid(x):
            return numpy_func(x)

    @_node_internal_use(name=name)
    def generic_type(x: ts["T"]) -> ts[generic_out_type]:
        if csp.valid(x):
            return op_lambda(x)

    def comp(x: ts["T"]):
        if get_origin(x.tstype.typ) in [Numpy1DArray, NumpyNDArray]:
            return numpy_type(x)
        elif x.tstype.typ is float:
            return float_type(x)
        elif x.tstype.typ is int:
            return int_type(x)
        return generic_type(x)

    comp.__name__ = name
    return comp


# Math operators
add = define_binary_op("add", lambda x, y: x + y)
sub = define_binary_op("sub", lambda x, y: x - y)
multiply = define_binary_op("multiply", lambda x, y: x * y)
divide = define_binary_op("divide", lambda x, y: x / y)
pow = define_binary_op("pow", lambda x, y: x**y)
min = define_binary_op("min", lambda x, y: x if x < y else y)
max = define_binary_op("max", lambda x, y: x if x > y else y)
floordiv = define_binary_op("floordiv", lambda x, y: x // y)
mod = define_binary_op("mod", lambda x, y: x % y)
pos = define_unary_op("pos", lambda x: +x)
neg = define_unary_op("neg", lambda x: -x)

# Because python's builtin abs is masked
# in the next definition, add a local
# variable so it can still be used in lambda.
# NOTE: this should not be exported in __all__
_python_abs = abs

# Other math ops
abs = define_unary_op("abs", lambda x: _python_abs(x))
ln = define_unary_op("ln", lambda x: math.log(x))
log2 = define_unary_op("log2", lambda x: math.log2(x))
log10 = define_unary_op("log10", lambda x: math.log10(x))
exp = define_unary_op("exp", lambda x: math.exp(x))
# could replace with math.exp2 once python3.10 and older aren't supported
exp2 = define_unary_op("exp2", lambda x: 2**x)
sqrt = define_unary_op("sqrt", lambda x: math.sqrt(x))
erf = define_unary_op("erf", lambda x: math.erf(x))
sin = define_unary_op("sin", lambda x: math.sin(x))
cos = define_unary_op("cos", lambda x: math.cos(x))
tan = define_unary_op("tan", lambda x: math.tan(x))
arcsin = define_unary_op("arcsin", lambda x: math.asin(x))
arccos = define_unary_op("arccos", lambda x: math.acos(x))
arctan = define_unary_op("arctan", lambda x: math.atan(x))
sinh = define_unary_op("sinh", lambda x: math.sinh(x))
cosh = define_unary_op("cosh", lambda x: math.cosh(x))
tanh = define_unary_op("tanh", lambda x: math.tanh(x))
arcsinh = define_unary_op("arcsinh", lambda x: math.asinh(x))
arccosh = define_unary_op("arccosh", lambda x: math.acosh(x))
arctanh = define_unary_op("arctanh", lambda x: math.atanh(x))

# Comparison operators
eq = define_binary_op("eq", lambda x, y: x == y)
ne = define_binary_op("ne", lambda x, y: x != y)
gt = define_binary_op("gt", lambda x, y: x > y)
lt = define_binary_op("lt", lambda x, y: x < y)
ge = define_binary_op("ge", lambda x, y: x >= y)
le = define_binary_op("le", lambda x, y: x <= y)
