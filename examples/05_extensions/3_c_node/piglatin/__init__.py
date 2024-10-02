import csp  # noqa: F401
from csp import ts

from . import _piglatin


@csp.node(cppimpl=_piglatin.piglatin)
def piglatin(x: ts[str], capitalize: bool = False) -> ts[str]:
    # Python implementation
    if csp.ticked(x):
        return "Its no fun if you don't use the C++ implementation!"
