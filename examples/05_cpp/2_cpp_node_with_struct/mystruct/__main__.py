from datetime import datetime

import csp

from .node import use_struct_generic, use_struct_specific
from .struct import MyStruct


@csp.graph
def my_graph() -> csp.Outputs(generic=csp.ts[MyStruct], specific=csp.ts[MyStruct]):
    st = csp.const(MyStruct(a=1, b="abc"))

    csp.print("input", st)
    generic = use_struct_generic(st)
    csp.print("use_struct_generic", generic)
    specific = use_struct_specific(generic)
    csp.print("use_struct_specific", specific)

    csp.output(generic=generic, specific=specific)


if __name__ == "__main__":
    start = datetime(2020, 1, 1)
    csp.run(my_graph, starttime=start)
