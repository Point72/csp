"""
A retail cart example that:
- Tracks the real-time total of a cart (as items are added and removed)
- Gives a 10% discount if the purchase is made within XX minutes
- Tracks the total sales and number of purchases
"""

from datetime import datetime, timedelta

import csp
import csp.showgraph
from csp import ts


PRODUCTS = {
    "X": 20.0,
    "Y": 50.0,
}


class Cart(csp.Struct):
    product: str
    qty: int
    add: bool
    purchase: bool


@csp.node
def track_discount(event: ts[Cart]) -> ts[float]:
    with csp.alarms():
        alarm = csp.alarm(bool)

    with csp.state():
        discount = 0.9

    with csp.start():
        csp.schedule_alarm(alarm, timedelta(seconds=5), True)

    if csp.ticked(alarm):
        discount = 1

    return discount


@csp.node
def update_cart(event: ts[Cart], discount: ts[float]) -> csp.Outputs(total = ts[float], items = ts[int]):
    with csp.state():
        cart_total = 0.0
        cart_items = 0

    if csp.ticked(event):
        if event.add:
            cart_total += PRODUCTS[event.product] * event.qty
            cart_items += event.qty
        else:
            cart_total -= PRODUCTS[event.product] * event.qty
            cart_items += event.qty

    final_total = cart_total * discount

    if event.purchase:
        cart_total = 0.0
        cart_items = 0

    csp.output(total=final_total, items=cart_items)


@csp.node
def track_purchases(event: ts[Cart], cart_total: ts[float]) -> csp.Outputs(sale = ts[float], qty = ts[int]):
    with csp.state():
        purchases = 0
        total_sales = 0.0

    purchases += 1
    total_sales += cart_total

    return csp.output(sale=total_sales, qty=purchases)


@csp.graph
def my_graph():
    st = datetime(2020, 1, 1)

    # Example cart updates
    events = csp.curve(
        Cart,
        [
            (st + timedelta(seconds=2), Cart(product="X", qty=1, add=True, purchase=False)),
            (st + timedelta(seconds=4), Cart(product="Y", qty=2, add=True, purchase=False)),
            (st + timedelta(seconds=6), Cart(product="Y", qty=1, add=False, purchase=True)),
            (st + timedelta(seconds=8), Cart(product="X", qty=1, add=True, purchase=False)),
        ],
    )

    csp.print("Events", events)

    discount = track_discount(events)
    current_cart = update_cart(events, discount)

    split_purchases = csp.split(events.purchase, events)
    purchases = track_purchases(split_purchases.false, current_cart.total)

    csp.print("Cart total", current_cart.total)
    csp.print("Cart items", current_cart.items)
    csp.print("Total sale", purchases.sale)
    csp.print("Purchases", purchases.qty)


def main():
    start = datetime(2020, 1, 1)
    csp.run(my_graph, starttime=start)
    csp.show_graph(my_graph, graph_filename="tmp.png")


if __name__ == "__main__":
    main()
