import csp
import csp.impl.pandas_accessor  ## This registers the "csp" accessors on pd.Series and pd.DataFrame
import numpy as np
import pandas as pd
import random
from csp.impl.pandas_ext_type import TsDtype
from csp.trading.examples.secmaster import ExampleSecMaster
from datetime import datetime, timedelta, date

import csp.impl.pandas_accessor  ## This registers the "csp" accessors on pd.Series and pd.DataFrame

## This example illustrates how csp edges can be used *inside* a standard pandas data frame via
## the pandas extension type mechanism

def run():

    secmaster = ExampleSecMaster()
    all_handles = secmaster.all_handles(date.today())

    # Create a static data frame
    N = 10
    td = timedelta(seconds=2)
    handles = [random.choice(all_handles) for _ in range(N)]
    df = pd.DataFrame({'ticker': [h.ticker() for h in handles],
                       'region': [h.region() for h in handles],
                       'exchange': [str(h.primary_exchange()) for h in handles],
                       'adv21': [h.adv21() for h in handles],
                       'ref_price': [h.ref_price() for h in handles],
                       },
                      index=[h.point_id() for h in handles])

    print('Create a standard dataframe...')
    print(df)
    print()
    print('Add some time series...')
    df['bid'] = pd.Series(
        [csp.timer(timedelta(seconds=random.choice([1, 2, 3])), 0.).apply(lambda x: (h.ref_price() - random.random()))
         for h in handles], index=df.index, dtype=TsDtype(float))
    df['ask'] = pd.Series(
        [csp.timer(timedelta(seconds=random.choice([1, 2, 3])), 0.).apply(lambda x: (h.ref_price() + random.random()))
         for h in handles], index=df.index, dtype=TsDtype(float))
    print(df)
    print()
    print('Compute a mid column...')
    df['mid'] = (df['bid'] + df['ask']) / 2.
    print(df)
    print()
    print('Notice the dtypes on the frame...')
    print(df.dtypes)
    print()
    print('Snap a "live" version of the frame...')
    print(df.csp.snap())
    print()
    print('Compute an edge for the weighted average price...')
    weights = np.array([random.randint(0, 10) for _ in handles])
    weighted_price = (df['mid']*weights).sum()/weights.sum()
    print(weighted_price)
    print()
    print('Run the weighted price...')
    print(weighted_price.run(starttime=datetime.utcnow(), endtime=timedelta(seconds=6)))
    print()
    # Numeric group-by aggregation is not supported on extension types until pandas version 1.3
    if pd.__version__ >= '1.3':
        print('Aggregate by exchange...')
        df_agg = df.groupby('exchange')['mid'].agg(['mean', 'sum'])
        print(df_agg)
        print()
        print('Run the frame as of now')
        df_agg.csp.run(datetime.utcnow(), timedelta(seconds=6))
        print(df)
        print()
    print()
    print()
    print()
    print('Convert the frame to a standard one')
    out = df.csp.run(datetime(2021, 1, 1), timedelta(seconds=10))
    print(out)
    print('Create a new data frame of csp columns by collapsing the bid and ask column into edges and applying '
          'an aggregation to the static columns')
    df2 = out.to_csp(columns=['bid', 'ask', 'mid'], agg='last')
    print(df2)
    print('Apply a non-csp function to transform all the edges in the column, and run it')
    out2 = df2['mid'].csp.apply(np.log).csp.run(starttime=datetime(2021, 1, 1), endtime=timedelta(seconds=10))
    print(out2)


if __name__ == '__main__':
    run()
