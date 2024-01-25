from datetime import datetime, timedelta

import csp
from csp import stats

"""
The csp.stats library contains numerous utilities for rolling window statistics. Full documentation: TO ADD
Below are some example use cases to show how single-input stats functions can be used
"""

# Example 1: Compute the following statistics for a single symbol
# 1) a 2-minute rolling VWAP starting at t=1 minute, with computation every 1 minute and data reset every 5 minutes
# 2) a EMA price with a 2-minute half-life, again computing once every minute with a reset every 5 minutes.
#    --- Additionally, we ensure that there are at least 2 data points for a computation
# 3) cumulative volume from the start of trading

st = datetime(2020, 1, 1)
prices_data = [
    (st + timedelta(minutes=1.3), 12.653),
    (st + timedelta(minutes=2.3), 14.210),
    (st + timedelta(minutes=3.8), 13.099),
    (st + timedelta(minutes=4.1), 12.892),
    (st + timedelta(minutes=4.4), 17.328),
    (st + timedelta(minutes=5.1), 18.543),
    (st + timedelta(minutes=5.3), 17.564),
    (st + timedelta(minutes=6.3), 19.023),
    (st + timedelta(minutes=8.7), 19.763),
]

volume_data = [
    (st + timedelta(minutes=1.3), 100),
    (st + timedelta(minutes=2.3), 115),
    (st + timedelta(minutes=3.8), 85),
    (st + timedelta(minutes=4.1), 90),
    (st + timedelta(minutes=4.4), 95),
    (st + timedelta(minutes=5.1), 185),
    (st + timedelta(minutes=5.3), 205),
    (st + timedelta(minutes=6.3), 70),
    (st + timedelta(minutes=8.7), 65),
]


def stats_graph():
    price = csp.curve(typ=float, data=prices_data)
    volume = csp.curve(typ=float, data=volume_data)

    # Trigger a computation every minute
    trigger = csp.timer(timedelta(minutes=1))

    # Reset the data every 5 minutes, but only after computing values
    reset = csp.delay(csp.timer(timedelta(minutes=5)), timedelta(microseconds=1))

    # calculate rolling 2 minute VWAP with our first data at 1 minute
    vwap = stats.mean(
        price,
        interval=timedelta(minutes=2),
        min_window=timedelta(minutes=1),
        trigger=trigger,
        weights=volume,
        reset=reset,
    )

    # calculate a time-weighted EMA of halflife 2 minutes for the price data
    ewm_price = stats.ema(price, halflife=timedelta(minutes=2), trigger=trigger, reset=reset, min_data_points=2)

    # calculate cumulative volume over the entire execution without reset
    total_vol = stats.sum(volume, interval=None, min_window=timedelta(minutes=1), trigger=trigger)

    csp.add_graph_output("vwap", vwap)
    csp.add_graph_output("ewm_price", ewm_price)
    csp.add_graph_output("total_vol", total_vol)


if __name__ == "__main__":
    results = csp.run(stats_graph, starttime=st, endtime=st + timedelta(minutes=10))

    for i in range(10):
        print(f"Time: {results['vwap'][i][0]}", end="\t")
        print(f"VWAP: {round(results['vwap'][i][1], 4)}", end="\t")
        print(f"Exp Price: {round(results['ewm_price'][i][1], 4)}", end="\t")
        print(f"Cum. Vol: {results['total_vol'][i][1]}")
