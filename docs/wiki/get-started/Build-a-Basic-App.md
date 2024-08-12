We have looked at the features of CSP nodes and graphs, as well as how to run an application using `csp.run`. In this tutorial, we will apply what we learned in [First Steps](First-Steps) and [More with CSP](More-with-CSP) to build a basic retail app which maintains an online shopping cart.
We will also introduce two important new concepts: the [`csp.Struct`](csp.Struct-API) data structure and multi-output nodes using `csp.Outputs`.

Our application will track a customer's shopping cart and apply a 10% discount for any items added to the cart in the first minute. Check out the complete code [here](examples/01_basics/e5_retail_cart.py).

## Structured data with `csp.Struct`

An individual item in a shopping cart consists of many fields; for example, the product's name, quantity and cost. The shopping cart itself may contain a list of these items as a field, plus a user ID or name. We also want to store updates to the shopping cart in an organized data structure, which has fields indicating the item in question and whether it was added or removed.

In `csp`, you can use a [`csp.Struct`](csp.Struct-API) to store typed fields together in a single data type. There are many advantages to using a `csp.Struct` instead of a standard Python dataclass. For example, the fields can be accessed as their own time series, ticking independently each time they update. Structs also have builtin conversion methods to JSON or dictionary objects. Due to their underlying C++ implementation, structs are also highly performant within `csp` compared to standard Python user-defined types.

```python
import csp
from typing import List

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
```

Any number of fields on a struct can be set by a user; others will remain unset with a special value of `csp.UNSET`. For example, when we remove an item in `CartUpdate`, the cost will not be set.

## Track cart updates

Recall from [More with CSP](More-with-CSP) that we can store state variables in a `csp.node` using a `csp.state` block. We will create a node that tracks updates to a user's cart by storing the `Cart` struct as a state variable named `s_cart`.

> \[!TIP\]
> By convention, state variables are prefixed with `s_` for readability.

A CSP node can return multiple named outputs. To annotate a multi-output node, we use `csp.Outputs` syntax for the return type annotation. To tick out each named value, we use the `csp.output` function. After each update event, we will tick out the total value of the user's cart and the number of items present.

To apply a discount for all items added in the first minute, we can use an alarm. We discussed how to use a `csp.alarm` as an internal time-series in the [Poisson counter example](More-with-CSP). We will only update the cart when the user adds, removes or purchases items. We need to know what the active discount rate to apply is but we don't need to trigger an update when it changes. To achieve this, we make the alarm time-series `discount` a *passive* input.

A *passive* input is a time-series input that will not cause the node to execute when it ticks. When we access the input within the node, we always get its most recent value. The opposite of passive inputs are *active* inputs, which trigger a node to compute upon a tick. So far, every input we've worked with has been an active input. We will set the discount input to be passive at graph startup.

> \[!TIP\]
> By default, all `csp.ts` inputs are active. You can change the activity of an input at any point during execution by using `csp.make_passive` or `csp.make_active`.

```python
from csp import ts
from datetime import timedelta
from functools import reduce

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
        csp.schedule_alarm(discount, timedelta(), 0.9) # 10% off for the first minute
        csp.schedule_alarm(discount, timedelta(minutes=1), 1.0) # full price after!

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
```

## Create workflow graph

To create example cart updates, we will use a [`csp.curve`](Base-Adapters-API#cspcurve) like we have in previous examples. The `csp.curve` replays a list of events at specific times.

```python
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
```

## Execute the graph

Execute the program and observe the outputs that our shopping cart provides.

```python
def main():
    csp.run(my_graph, starttime=st)
```

```raw
2020-01-01 00:00:15 Events:CartUpdate( item=Item( name=X, cost=10.0, qty=1 ), add=True )
2020-01-01 00:00:15 Cart total:9.0
2020-01-01 00:00:15 Cart number of items:1
2020-01-01 00:00:30 Events:CartUpdate( item=Item( name=Y, cost=15.0, qty=2 ), add=True )
2020-01-01 00:00:30 Cart total:36.0
2020-01-01 00:00:30 Cart number of items:3
2020-01-01 00:00:45 Events:CartUpdate( item=Item( name=Y, cost=<unset>, qty=1 ), add=False )
2020-01-01 00:00:45 Cart total:22.5
2020-01-01 00:00:45 Cart number of items:2
2020-01-01 00:01:15 Events:CartUpdate( item=Item( name=Z, cost=20.0, qty=1 ), add=True )
2020-01-01 00:01:15 Cart total:42.5
2020-01-01 00:01:15 Cart number of items:3
```
