from datetime import datetime, timedelta

import numpy as np

import csp

"""
The csp.stats library contains numerous utilities for rolling window statistics. Full documentation: TO ADD
Below are some example use cases to show how NumPy arrays can be used for efficient element-wise stats computations
"""

# Example 2: Compute the following statistics for a set of three symbols
# 1) 5-minute rolling mean and geometric mean of each symbol's price computed every minute, starting at 1 minute
# 2) an adjusted EMA of each symbol's price, also computed once a minute
# 3) rolling 5-minute correlation matrix between all 3 symbols, beginning at 3-minutes

st = datetime(2020, 1, 1)

# Prices are sampled every 1 minute
symb1_prices = [8.65, 8.67, 8.72, 8.68, 8.90, 9.1, 9.04, 9.11, 9.34, 9.36]
symb2_prices = [314.34, 315.67, 316.70, 316.45, 320.10, 323.84, 322.76, 328.56, 328.60, 329.60]
symb3_prices = [23.75, 23.55, 23.23, 23.98, 22.10, 21.89, 21.78, 20.50, 21.00, 21.23]

prices_data = [
    (st + timedelta(minutes=i + 1), np.array([symb1_prices[i], symb2_prices[i], symb3_prices[i]], dtype=float))
    for i in range(10)
]


@csp.graph
def numpy_stats_graph():
    price = csp.curve(typ=np.ndarray, data=prices_data)

    # Trigger a computation every minute
    trigger = csp.timer(timedelta(minutes=1))

    # calculate rolling 5 minute mean price for each symbol, starting at 1 minute of data
    avg_price = csp.stats.mean(price, interval=timedelta(minutes=5), min_window=timedelta(minutes=1), trigger=trigger)

    # calculate rolling 5 minute geometric mean price for each symbol
    geom_avg_price = csp.stats.gmean(
        price, interval=timedelta(minutes=5), min_window=timedelta(minutes=1), trigger=trigger
    )

    # calculate an adjusted EMA for the prices
    ewm_price = csp.stats.ema(price, alpha=0.1, adjust=True, trigger=trigger)

    # calculate 5 minute correlation between all 3 symbols, starting at 3 minutes
    corr_matrix = csp.stats.corr_matrix(
        price, interval=timedelta(minutes=5), min_window=timedelta(minutes=3), trigger=trigger
    )

    csp.add_graph_output("avg_price", avg_price)
    csp.add_graph_output("geom_avg_price", geom_avg_price)
    csp.add_graph_output("ewm_price", ewm_price)
    csp.add_graph_output("corr_matrix", corr_matrix)


def main():
    results = csp.run(numpy_stats_graph, starttime=st, endtime=st + timedelta(minutes=10), realtime=False)

    print("Price Averages\n")
    for i in range(10):
        print(f"Time: {results['avg_price'][i][0]}", end="\t")
        print(f"Mean Price: {results['avg_price'][i][1]}", end="\t")
        print(f"Geom. Mean Price: {results['geom_avg_price'][i][1]}", end="\t")
        print(f"Exp. Price: {results['ewm_price'][i][1]}")

    print("\nCorrelation\n")
    for i in range(7):
        print(f"Time: {results['corr_matrix'][i][0]}", end="\t")
        print(f"Corr. Matrix:\n{results['corr_matrix'][i][1]}")


if __name__ == "__main__":
    main()
