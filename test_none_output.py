import csp
from datetime import datetime

# Test case 1: Node with None output annotation
@csp.node
def n(x: csp.ts[int]) -> None:
    print(x)

# Test case 2: Graph with None output annotation
@csp.graph
def g() -> None:
    n(csp.const(1))

csp.run(g, datetime(2020, 1, 1))
