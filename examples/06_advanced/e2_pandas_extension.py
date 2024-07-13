"""
This example illustrates how csp edges can be used inside a pandas data frame via the pandas extension type mechanism.
"""

import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import csp
import csp.impl.pandas_accessor  # This registers the "csp" accessors on pd.Series and pd.DataFrame
from csp.impl.pandas_ext_type import TsDtype
from csp.random import brownian_motion
from csp.stats import numpy_to_list


def main():
    random.seed(1234)
    rng = np.random.default_rng(seed=None)
    N = 10
    symbols = [f"SYMBOL_{i}" for i in range(N)]
    regions = [random.choice(["US", "EU", "AP"]) for _ in range(N)]
    exchanges = [region + random.choice(["X", "Y"]) for region in regions]
    open_prices = np.round(np.random.rand(N) * 200, 2)

    # Create a static data frame
    df = pd.DataFrame(
        {
            "region": regions,
            "exchange": exchanges,
            "open_price": open_prices,
        },
        index=symbols,
    )

    print("Create a standard dataframe...")
    print(df)
    print()
    print("Add some time series...")
    trigger = csp.timer(timedelta(seconds=2))
    mids = (
        brownian_motion(
            trigger, drift=csp.const(np.zeros(N)), covariance=csp.const(0.01 * np.diag(np.ones(N))), seed=rng
        ).apply(np.exp)
        * open_prices
    )  # Ignore drift adjustment for simplicity
    df["mid"] = pd.Series(numpy_to_list(mids, len(df.index)), index=df.index, dtype=TsDtype(float))
    print(df)
    print()
    print("Compute bid and ask columns...")
    width = csp.const(0.25)
    df["bid"] = df["mid"] - width / 2.0
    df["ask"] = df["mid"] + width / 2.0
    print(df)
    print()
    print("Notice the dtypes on the frame...")
    print(df.dtypes)
    print()
    print('Snap a "live" version of the frame...')
    print(df.csp.snap())
    print()
    print("Compute an edge for the weighted average price...")
    weights = np.array([random.randint(0, 10) for _ in symbols])
    weighted_price = (df["mid"] * weights).sum() / weights.sum()
    print(weighted_price)
    print()
    print("Run the weighted price as a graph...")
    for timestamp, value in weighted_price.run(starttime=datetime.utcnow(), endtime=timedelta(seconds=6)):
        print(timestamp, value)
    print()
    print()
    # Numeric group-by aggregation is not supported on extension types until pandas version 1.3
    if pd.__version__ >= "1.3":
        print("Aggregate by exchange (mix of csp and non-csp results)...")
        df_agg = df.groupby("exchange")["mid"].agg(["count", "mean", "sum"])
        print(df_agg)
        print()
        print("Run the aggregate frame as of now")
        print(df_agg.csp.run(datetime.utcnow(), timedelta(seconds=6)))
        print()
    print()
    print()
    print("Convert the original frame to a standard pandas frame with an extra index for the time dimension")
    out = df.csp.run(datetime(2024, 1, 1), timedelta(seconds=10))
    print(out)
    print(
        "Convert the above result back to the original dataframe by turning the mid, bid, and ask column into edges and applying "
        'a "last" aggregation to the static columns'
    )
    df2 = out.to_csp(columns=["bid", "ask", "mid"], agg="last")
    print(df2)
    print()
    print('Apply a non-csp function (i.e. np.log) to transform all the edges in the "mid" column, and run it')
    out2 = df2["mid"].csp.apply(np.log).csp.run(starttime=datetime(2024, 1, 1), endtime=timedelta(seconds=10))
    print(out2)


if __name__ == "__main__":
    main()
