csp offers several built-in generic, mathematical, and statistical nodes that are often required in streaming workflows. This allows you to write applications quickly, and update and expand them as required by including new nodes.

In this tutorial, you will calculate the [volume weighted average price (VWAP)](https://www.investopedia.com/terms/v/vwap.asp) and the profit and loss (PnL) of the trade, on a stream of trades. Check out the complete example: [trade profit-and-loss example](examples/01_basics/e4_trade_pnl.py)

## Compound inputs with `csp.Struct`

A Trade consists of multiple parts: price of a share, quantity of shares, and indication of a buying or a selling transaction.

`struct` is a useful data type for this type of information. In csp, you can use [`csp.Struct`](csp.Struct-API), a higher-level data type that can be defined with type-annotated values.

```python
class Trade(csp.Struct):
    price: float
    qty: int
    buy: bool
```

## Build computing nodes

To calculate volume-weighted averages, you need the cumulative sum of previous trade prices and quantities. Hence, your csp node needs to store stateful information. You can use \[`csp.state`\] to declare stateful variables that bound to the node.

A csp node can return multiple named outputs, denoted as `csp.Outputs` type, created using the `csp.output` function. The individual return values can be accessed with dot notation.

```python
@csp.node
def vwap(trade: ts[Trade]) -> csp.Outputs(vwap=ts[float], qty=ts[int]):
    with csp.state():
        s_cum_notional = 0.0
        s_cum_qty = 0

    if csp.ticked(trade):
        s_cum_notional += trade.price * trade.qty
        s_cum_qty += trade.qty

        csp.output(vwap=s_cum_notional / s_cum_qty, qty=s_cum_qty)
```

```python
@csp.node
def calc_pnl(vwap_trade: ts[Trade], mark_price: ts[float]) -> ts[float]:
    if csp.ticked(vwap_trade, mark_price) and csp.valid(vwap_trade, mark_price):
        if vwap_trade.buy:
            pnl = (mark_price - vwap_trade.price) * vwap_trade.qty
        else:
            pnl = (vwap_trade.price - mark_price) * vwap_trade.qty

        return pnl
```

## Create workflow graph

To create example `ask`, `bid`, and `trades` values, you can use [`csp.curve`](Base-Adapters-API#cspcurve).
This commonly-used type in csp converts a list of (non-CSP) data into a ticking, csp-friendly inputs.

```python
@csp.graph
def my_graph():
    st = datetime(2020, 1, 1)

    # Dummy bid/ask trade inputs
    bid = csp.curve(
        float,
        [(st + timedelta(seconds=0.5), 99.0), (st + timedelta(seconds=1.5), 99.1), (st + timedelta(seconds=5), 99.2)],
    )

    ask = csp.curve(
        float,
        [
            (st + timedelta(seconds=0.6), 99.1),
            (st + timedelta(seconds=1.3), 99.2),
            (st + timedelta(seconds=4.2), 99.25),
        ],
    )

    trades = csp.curve(
        Trade,
        [
            (st + timedelta(seconds=1), Trade(price=100.0, qty=50, buy=True)),
            (st + timedelta(seconds=2), Trade(price=101.5, qty=500, buy=False)),
            (st + timedelta(seconds=3), Trade(price=100.50, qty=100, buy=True)),
            (st + timedelta(seconds=4), Trade(price=101.2, qty=500, buy=False)),
            (st + timedelta(seconds=5), Trade(price=101.3, qty=500, buy=False)),
            (st + timedelta(seconds=6), Trade(price=101.4, qty=500, buy=True)),
        ],
    )
```

The next step is to separate the buying and selling transactions that are captured in `Trade.buy` and calculate the VWAP. You can use the [csp.split](Base-Nodes-API#cspsplit) function for this. It splits input based on a boolean flag.

> \[!TIP\]
> To perform the opposite operation of a split you can use [csp.merge](Base-Nodes-API#cspmerge).

```python
buysell = csp.split(trades.buy, trades)
buy_trades = buysell.true
sell_trades = buysell.false

buy_vwap = vwap(buy_trades)
sell_vwap = vwap(sell_trades)
```

Finally, you need to calculate the profit-and-loss using the VWAPs of trades. You can create new "Trade" `Struct`s with  using [`fromts`](csp.Struct-API#available-methods) and perform the computation:

```python
buy_vwap = Trade.fromts(price=buy_vwap.vwap, qty=buy_vwap.qty, buy=buy_trades.buy)
sell_vwap = Trade.fromts(price=sell_vwap.vwap, qty=sell_vwap.qty, buy=sell_trades.buy)

mid = (bid + ask) / 2
buy_pnl = calc_pnl(buy_vwap, mid)
sell_pnl = calc_pnl(sell_vwap, mid)

pnl = buy_pnl + sell_pnl

 csp.print("buys", buy_trades)
csp.print("sells", sell_trades)
csp.print("buy_vwap", buy_vwap)
csp.print("sell_vwap", sell_vwap)

csp.print("mid", mid)
csp.print("buy_pnl", buy_pnl)
csp.print("sell_pnl", sell_pnl)
csp.print("pnl", pnl)
```

Execute the program and generate an image of the graph with:

```python
def main():
    start = datetime(2020, 1, 1)
    csp.run(my_graph, starttime=start, endtime=timedelta(seconds=20))
    csp.show_graph(my_graph, graph_filename="tmp.png")

if __name__ == "__main__":
    main()
```

As expected, the graph for this workflow show the split of buy & sell Trades, calculation of VWAP for each followed by using the VWAP-Stucts in profit-and-loss calculations:

![Output of show_graph](images/pnl-graph.png)

Check out the [trade profit-and-loss example](https://github.com/Point72/csp/blob/main/examples/01_basics/e4_trade_pnl.py) to learn more.
