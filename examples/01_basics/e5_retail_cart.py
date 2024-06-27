"""
A retail cart example that:
- Tracks the real-time total of a cart (as items are added and removed)
- Applies a 10% discount if the purchase is made within 1 minute
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
    """
    Track the discount based on the time of the purchase.
    If the purchase is made within 1 minute, a 10% discount is applied.
    """
    with csp.alarms():
        alarm = csp.alarm(bool)

    with csp.state():
        s_discount = 0.9  # 10% discount

    with csp.start():
        csp.schedule_alarm(alarm, timedelta(seconds=60), True)

    if csp.ticked(alarm):
        s_discount = 1.0  # No discount

    if csp.ticked(event):
        return s_discount


@csp.node
def update_cart(event: ts[Cart], discount: ts[float]) -> csp.Outputs(total = ts[float], items = ts[int]):
    """
    Track of the cart total and number of items.
    """
    with csp.state():
        s_cart_total = 0.0
        s_cart_items = 0

    if csp.ticked(event):
        if event.add:
            s_cart_total += PRODUCTS[event.product] * event.qty
            s_cart_items += event.qty
        else:
            s_cart_total -= PRODUCTS[event.product] * event.qty
            s_cart_items += event.qty

    final_total = s_cart_total * discount

    if event.purchase:
        s_cart_total = 0.0
        s_cart_items = 0

    csp.output(total=final_total, items=s_cart_items)


@csp.node
def track_purchases(purchase_event: ts[Cart], cart_total: ts[float]) -> csp.Outputs(sale = ts[float], qty = ts[int]):
    """
    Track the total sales and number of purchases.
    """
    with csp.state():
        s_purchases = 0
        s_total_sales = 0.0

    if csp.ticked(purchase_event):
        s_purchases += 1
        s_total_sales += cart_total

    return csp.output(sale=s_total_sales, qty=s_purchases)


@csp.graph
def my_graph():
    st = datetime(2020, 1, 1)

    # Example cart updates
    events = csp.curve(
        Cart,
        [
            (st + timedelta(seconds=15), Cart(product="X", qty=1, add=True, purchase=False)),
            (st + timedelta(seconds=30), Cart(product="Y", qty=2, add=True, purchase=False)),
            (st + timedelta(seconds=40), Cart(product="Y", qty=1, add=False, purchase=True)),
            (st + timedelta(seconds=55), Cart(product="X", qty=1, add=True, purchase=False)),
            (st + timedelta(seconds=90), Cart(product="X", qty=1, add=True, purchase=True)),
        ],
    )

    csp.print("Events", events)

    discount = track_discount(events)
    current_cart = update_cart(events, discount)

    # Split the purchase=True events to send to the track_purchases node
    split_purchases = csp.split(events.purchase, events)
    purchases = track_purchases(split_purchases.true, current_cart.total)

    csp.print("Cart items", current_cart.items)
    csp.print("Cart total", current_cart.total)
    csp.print("Discount %", (1.0-discount)*100)
    csp.print("Total sales", purchases.sale)
    csp.print("Purchases", purchases.qty)


def main():
    start = datetime(2020, 1, 1)
    csp.run(my_graph, starttime=start)
    csp.show_graph(my_graph, graph_filename="tmp.png")


if __name__ == "__main__":
    main()
