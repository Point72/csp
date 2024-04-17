import csp  # noqa: F401
from csp import ts

from . import _mystruct
from .struct import MyStruct


@csp.node(cppimpl=_mystruct.use_struct_generic)
def use_struct_generic(x: ts[MyStruct]) -> ts[MyStruct]:
    # Python implementation
    if csp.ticked(x):
        print("Its no fun if you don't use the C++ implementation!")
        return x


@csp.node(cppimpl=_mystruct.use_struct_specific)
def use_struct_specific(x: ts[MyStruct]) -> ts[MyStruct]:
    # Python implementation
    if csp.ticked(x):
        print("Its no fun if you don't use the C++ implementation!")
        return x
