CSP offers several built-in generic, mathematical, and statistical nodes that are often required in streaming workflows. This allows you to write applications quickly, and update and expand them as required by including new nodes.

In this tutorial, you build a CSP program to track a sample shopping cart and purchases, where a 10% discount if the purchase is made within 1 minute. You will learn to create and use composite data types, and add more functionality to nodes.

Check out the complete example: [Retail cart example](examples/01_basics/e5_retail_cart.py).

## Compound inputs with `csp.Struct`

A Cart consists of multiple parts: product name, quantity, indication if the product was added or removed from the cart, and if a purchase was made.

A `struct` (in the C programming language) is a useful data type for this type of information. In CSP, you can use [`csp.Struct`](csp.Struct-API), a higher-level data type that can be defined with type-annotated values.

```python
class Cart(csp.Struct):
    product: str
    qty: int
    add: bool
    purchase: bool
```

## Build computing nodes

### Track cart updates

To track the total cart items and price, you need stateful variables that can be updated each time the cart is modified.
You can use `csp.state` to declare stateful variables that are bound to a node.

> \[!TIP\]
> By convention, state variables are prefixed with `s_`.

A CSP node can also return multiple named outputs, denoted as `csp.Outputs` type, created using the `csp.output` function. The individual return values can then be accessed with dot notation.

```python
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
```

### Track discount applied

To apply a discount if the purchase is made in under a minute, you need to keep track of time from the first update to the cart.

You can do this with an alarm in CSP. An alarm is a new Time Series input bound to a node that ticks at scheduled intervals. It can be created using `csp.alarm()` within an `csp.alarms()` context.

```python
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
```

### Track purchases

You can use stateful variables again to keep track of purchases.

> \[!NOTE\]
> While this information can also be tracked in `update_cart`, we create this independent node for
> composability and to introduce the concept of splitting the event stream in the following section.

```python
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
```

## Create workflow graph

To create example cart updates, you can use [`csp.curve`](Base-Adapters-API#cspcurve).
This commonly-used type in csp converts a list of (non-CSP) data into a ticking, csp-friendly inputs.

```python
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
```

To track total purchases, you only need the events where `purchase=True`.
You can use the [`csp.split`](https://github.com/Point72/csp/wiki/Base-Nodes-API#cspsplit) function for this. It splits the input based on a boolean flag.

```python
split_purchases = csp.split(events.purchase, events)
purchases = track_purchases(split_purchases.true, current_cart.total)
```

> \[!TIP\]
> To perform the opposite operation of a split you can use [csp.merge](Base-Nodes-API#cspmerge).

Finally print all the values.

```python
csp.print("Cart items", current_cart.items)
csp.print("Cart total", current_cart.total)
csp.print("Discount", discount)
csp.print("Total sales", purchases.sale)
csp.print("Purchases", purchases.qty)
```

## Execute the graph

Execute the program and generate an image of the graph with `csp.run` and `csp.show_graph` respectively:

```python
def main():
    start = datetime(2020, 1, 1)
    csp.run(my_graph, starttime=start)
    csp.show_graph(my_graph, graph_filename="tmp.png")
```

As expected, the graph for the workflow shows the three computing nodes as well the split on purchases.

![Output of show_graph](images/retail-graph.png)
