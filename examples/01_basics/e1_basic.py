from datetime import datetime

import csp
from csp import ts


@csp.node
def add(x: ts[int], y: ts[int]) -> ts[int]:
    return x + y


@csp.graph
def my_graph():
    x = csp.const(1)
    y = csp.const(2)

    sum = add(x, y)

    csp.print("x", x)
    csp.print("y", y)
    csp.print("sum", sum)


def main():
    csp.run(my_graph, starttime=datetime.now())


if __name__ == "__main__":
    main()
