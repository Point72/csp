from datetime import datetime, timedelta

import csp
from csp import ts


@csp.node
def spread(bid: ts[float], ask: ts[float]) -> ts[float]:
    if csp.ticked(bid, ask) and csp.valid(bid, ask):
        return ask - bid


@csp.graph
def my_graph():
    bid = csp.count(csp.timer(timedelta(seconds=2.5), True))
    ask = csp.count(csp.timer(timedelta(seconds=1), True))
    bid = bid * 2.0
    ask = ask * 2.0
    s1 = spread(bid, ask)
    s2 = ask - bid

    csp.print("bid", bid)
    csp.print("ask", ask)
    csp.print("spread", s1)
    csp.print("spread2", s2)


if __name__ == "__main__":
    csp.show_graph(my_graph)
    realtime = True
    if realtime:
        csp.run(my_graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=True)
    else:
        csp.run(my_graph, starttime=datetime(2020, 3, 1), endtime=timedelta(seconds=5))
