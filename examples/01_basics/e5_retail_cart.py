"""
A retail cart example that:
- Tracks the real-time total of a cart (as items are added and removed)
- Applies a 10% discount if the purchase is made within 1 minute
- Tracks the total sales and number of purchases
"""

from datetime import datetime, timedelta
from functools import reduce
from typing import List

import csp
from csp import ts


class Item(csp.Struct):
    name: str
    cost: float
    qty: int


class Cart(csp.Struct):
    user_id: int
    items: List[Item]


class CartUpdate(csp.Struct):
    item: Item
    add: bool


@csp.node
def update_cart(event: ts[CartUpdate], user_id: int) -> csp.Outputs(total=ts[float], num_items=ts[int]):
    """
    Track of the cart total and number of items.
    """
    with csp.alarms():
        discount = csp.alarm(float)

    with csp.state():
        # create an empty shopping cart
        s_cart = Cart(user_id=user_id, items=[])

    with csp.start():
        csp.make_passive(discount)
        csp.schedule_alarm(discount, timedelta(), 0.9)  # 10% off for the first minute
        csp.schedule_alarm(discount, timedelta(minutes=1), 1.0)  # full price after!

    if csp.ticked(event):
        if event.add:
            # apply current discount
            event.item.cost *= discount
            s_cart.items.append(event.item)
        else:
            # remove the given qty of the item
            new_items = []
            remaining_qty = event.item.qty
            for item in s_cart.items:
                if item.name == event.item.name:
                    if item.qty > remaining_qty:
                        item.qty -= remaining_qty
                        new_items.append(item)
                    else:
                        remaining_qty -= item.qty
                else:
                    new_items.append(item)
            s_cart.items = new_items

    current_total = reduce(lambda a, b: a + b.cost * b.qty, s_cart.items, 0)
    current_num_items = reduce(lambda a, b: a + b.qty, s_cart.items, 0)
    csp.output(total=current_total, num_items=current_num_items)


st = datetime(2020, 1, 1)


@csp.graph
def my_graph():
    # Example cart updates
    events = csp.curve(
        CartUpdate,
        [
            # Add 1 unit of X at $10 plus a 10% discount
            (st + timedelta(seconds=15), CartUpdate(item=Item(name="X", cost=10, qty=1), add=True)),
            # Add 2 units of Y at $15 each, plus a 10% discount
            (st + timedelta(seconds=30), CartUpdate(item=Item(name="Y", cost=15, qty=2), add=True)),
            # Remove 1 unit of Y
            (st + timedelta(seconds=45), CartUpdate(item=Item(name="Y", qty=1), add=False)),
            # Add 1 unit of Z at $20 but no discount, since our minute expired
            (st + timedelta(seconds=75), CartUpdate(item=Item(name="Z", cost=20, qty=1), add=True)),
        ],
    )

    csp.print("Events", events)

    current_cart = update_cart(events, user_id=42)

    csp.print("Cart number of items", current_cart.num_items)
    csp.print("Cart total", current_cart.total)


def main():
    csp.run(my_graph, starttime=st)


if __name__ == "__main__":
    main()
