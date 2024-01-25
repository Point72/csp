from datetime import datetime, timedelta

import csp
import csp.showgraph
from csp import ts


class Trade(csp.Struct):
    price: float
    qty: int
    buy: bool


@csp.node
def vwap(trade: ts[Trade]) -> csp.Outputs(vwap=ts[float], qty=ts[int]):
    with csp.state():
        s_cum_notional = 0.0
        s_cum_qty = 0

    if csp.ticked(trade):
        s_cum_notional += trade.price * trade.qty
        s_cum_qty += trade.qty

        csp.output(vwap=s_cum_notional / s_cum_qty, qty=s_cum_qty)


@csp.node
def calc_pnl(vwap_trade: ts[Trade], mark_price: ts[float]) -> ts[float]:
    if csp.ticked(vwap_trade, mark_price) and csp.valid(vwap_trade, mark_price):
        if vwap_trade.buy:
            pnl = (mark_price - vwap_trade.price) * vwap_trade.qty
        else:
            pnl = (vwap_trade.price - mark_price) * vwap_trade.qty

        return pnl


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
    buysell = csp.split(trades.buy, trades)
    buy_trades = buysell.true
    sell_trades = buysell.false

    buy_vwap = vwap(buy_trades)
    sell_vwap = vwap(sell_trades)

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


def main():
    start = datetime(2020, 1, 1)
    show_graph = False
    if show_graph:
        csp.showgraph.show_graph(my_graph)
    else:
        csp.run(my_graph, starttime=start)


if __name__ == "__main__":
    main()
