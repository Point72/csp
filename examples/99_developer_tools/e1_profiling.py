from datetime import datetime, timedelta

import csp
from csp import profiler

st = datetime(2020, 1, 1)


@csp.graph
def graph1():
    x = csp.curve(int, [(st + timedelta(seconds=i), i) for i in range(100)])  # 1,2,3...100
    y = x**2

    z = x + y
    w = csp.merge(y, z)
    p = csp.merge(x, w)
    o = csp.merge(w, p)

    csp.add_graph_output("o", o)


def main():
    # Example 1: view a graph's static attributes using graph_info
    info = profiler.graph_info(graph1)  # noqa: F841

    # Uncomment line below to print only the static graph info for graph1
    # info.print_info()

    # Example 2: profile a graph in runtime

    with profiler.Profiler() as p:
        csp.run(graph1, starttime=st, endtime=st + timedelta(seconds=100))

    prof = p.results()
    prof.print_stats()


if __name__ == "__main__":
    main()
