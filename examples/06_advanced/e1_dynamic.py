from datetime import datetime, timedelta

import csp
from csp import ts

# This example demonstrates the advanced concept of dynamic graphs. Dynamic graphs provide the ability to extend the shape of the graph during runtime,
# which is useful when you may not necessarily know what you will be processing at start


class Order(csp.Struct):
    symbol: str
    size: int
    price: float


@csp.graph
def process_symbol(symbol: str, order: ts[Order], initial_order: Order, timer: ts[int], scalar: str) -> ts[int]:
    print("Starting sub-graph to process symbol ", symbol, " with initial order: ", initial_order, " scalar: ", scalar)

    csp.print(symbol + " orders", order)
    csp.print(symbol + " timer", timer)

    cum_size = csp.accum(order.size)
    return cum_size


@csp.node
def process_results(x: {ts[str]: ts[int]}):
    if csp.ticked(x):
        print(csp.now(), "cum_sizes:", dict(x.tickeditems()))


@csp.graph
def main_graph():
    # We have a stream of incoming orders to deal with, we dont know the symbols up front
    orders = csp.curve(
        Order,
        [
            (timedelta(seconds=0), Order(symbol="AAPL", price=135, size=100)),
            (timedelta(seconds=1), Order(symbol="FB", price=350, size=-200)),
            (timedelta(seconds=2), Order(symbol="GME", price=210, size=1000)),
            (timedelta(seconds=3), Order(symbol="AAPL", price=138, size=-100)),
            (timedelta(seconds=4), Order(symbol="FB", price=330, size=100)),
            (timedelta(seconds=5), Order(symbol="AMC", price=57, size=400)),
            (timedelta(seconds=6), Order(symbol="GME", price=200, size=800)),
        ],
    )

    # Get a dynamic basket keys by symbol
    trigger = csp.dynamic_demultiplex(orders, orders.symbol)

    some_ts = csp.count(csp.timer(timedelta(seconds=1)))
    some_scalar = "howdy"

    # dynamic graphs
    cum_sizes = csp.dynamic(
        trigger,
        process_symbol,
        csp.snapkey(),  # csp.snapkey() provides the key that triggers a new dynamic graph as a scalar argument
        csp.attach(),  # csp.attach() will pass the corresponding timeseries of the key for the graph instance
        csp.snap(
            orders
        ),  # csp.snap will provide the current value of the given timeseries at the time of dynamic graph instantiation
        some_ts,  # regular time series can be passed along, which will be shared across all instances
        some_scalar,  # regular scalar values can be passed as arguments to the sub-graph as well
    )

    # cum_sizes is a dynamic basket of results, keyed by the trigger keys
    process_results(cum_sizes)


def main():
    csp.run(main_graph, starttime=datetime.utcnow().replace(microsecond=0), endtime=timedelta(seconds=10))


if __name__ == "__main__":
    main()
