#
# The csp.feedback construct is used to introduce cycles in the acyclical graph
#
from datetime import timedelta

import csp
import csp.showgraph
from csp import ts
from csp.utils.datetime import utc_now


class Order(csp.Struct):
    order_id: int
    price: float
    qty: int
    side: str


class ExecReport(csp.Struct):
    order_id: int
    status: str


# Simulate acking an order
@csp.node
def my_exchange(order: ts[Order]) -> ts[ExecReport]:
    with csp.alarms():
        delayed_order = csp.alarm(Order)

    with csp.state():
        s_delay = timedelta(seconds=0.7)

    if csp.ticked(order):
        csp.schedule_alarm(delayed_order, s_delay, order)

    if csp.ticked(delayed_order):
        return ExecReport(order_id=delayed_order.order_id, status="ACK")


@csp.node
def my_algo(exec_report: ts[ExecReport]) -> ts[Order]:
    with csp.alarms():
        new_order = csp.alarm(bool)

    with csp.state():
        s_lastprice = 100.0
        s_lastid = 1

    with csp.start():
        csp.schedule_alarm(new_order, timedelta(), True)

    if csp.ticked(new_order):
        order = Order(order_id=s_lastid, price=s_lastprice, qty=200, side="BUY")
        s_lastid += 1
        s_lastprice += 0.01
        csp.schedule_alarm(new_order, timedelta(seconds=1), True)
        print(f"{csp.now()} Sending new order id:{order.order_id} price {order.price}")
        return order

    if csp.ticked(exec_report):
        print(csp.now(), exec_report)


@csp.graph
def my_graph():
    # create the feedback first so that we can refer to it later
    exec_report_fb = csp.feedback(ExecReport)

    # generate "orders"
    orders = my_algo(exec_report_fb.out())

    # get exec_reports from "simulator"
    exec_report = my_exchange(orders)

    # now bind the exec reports to the feedback, finishing the "loop"
    exec_report_fb.bind(exec_report)


def main():
    show_graph = False
    if show_graph:
        csp.showgraph.show_graph(my_graph)
    else:
        csp.run(my_graph, starttime=utc_now(), endtime=timedelta(seconds=5), realtime=False)


if __name__ == "__main__":
    main()
