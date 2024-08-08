from datetime import datetime, timedelta

import csp
from csp import ts


@csp.node
def add(x: ts[int], y: ts[int]) -> ts[int]:
    return x + y


@csp.node
def accum(val: ts[int]) -> ts[int]:
    with csp.state():
        s_sum = 0
    if csp.ticked(val):
        s_sum += val
        return s_sum


@csp.graph
def my_graph():
    st = datetime(2020, 1, 1)

    # Dummy x values
    x = csp.curve(int, [(st + timedelta(1), 1), (st + timedelta(2), 2), (st + timedelta(3), 3)])

    # Dummy y values
    y = csp.curve(int, [(st + timedelta(1), -1), (st + timedelta(3), -1), (st + timedelta(4), -1)])

    # Add the time series
    sum = add(x, y)

    # Accumulate the result
    acc = accum(sum)

    csp.print("x", x)
    csp.print("y", y)
    csp.print("sum", sum)
    csp.print("accum", acc)


def main():
    start = datetime(2020, 1, 1)
    csp.run(my_graph, starttime=start)


if __name__ == "__main__":
    main()
