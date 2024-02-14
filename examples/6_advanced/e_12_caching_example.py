import math
import os
import pyarrow
from datetime import datetime, timedelta
from random import randint, random

import csp
from csp import Config, ts
from csp.cache_support import CacheConfig, CacheConfigResolver, GraphCacheOptions
from csp.impl.config import CacheCategoryConfig


class Book(csp.Struct):
    exchange_timestamp: datetime
    bid: float
    bid_size: int
    ask: float
    ask_size: float


@csp.node
def book(ticker: str) -> ts[Book]:
    with csp.alarms():
        a = csp.alarm(bool)

    with csp.state():
        mid = float
        spread = float
        mid = hash(ticker) % 100

    with csp.start():
        csp.schedule_alarm(a, timedelta(minutes=1), False)

    if csp.ticked(a):
        csp.schedule_alarm(a, timedelta(minutes=1), False)
        mid = mid + random()
        spread = random()
        bid = max(0, mid - spread / 2)
        ask = mid + spread / 2
        bid_size = randint(100, 10000)
        ask_size = randint(100, 10000)
        return Book(
            exchange_timestamp=csp.now() - timedelta(milliseconds=200),
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
        )


@csp.graph
def mid(ticker: str) -> csp.ts[float]:
    @csp.node
    def _calc(book: csp.ts[Book]) -> csp.ts[float]:
        if csp.ticked(book):
            return (book.ask + book.bid) / 2

    return _calc(book(ticker))


@csp.node
def average_spread(book: ts[Book], decay_factor: float = 1) -> ts[float]:
    with csp.state():
        last_time = float
        cur_average = float
        last_time = None
        cur_average = None

    if csp.ticked(book):
        spread = book.ask - book.bid
        cur_time = csp.now()
        if last_time is None:
            cur_average = spread
        else:
            q = math.exp(-decay_factor * (cur_time - last_time).total_seconds())
            cur_average = cur_average * q + (1 - q) * spread

        last_time = cur_time
        return cur_average


@csp.graph(cache=True)
def get_book_with_average_spread(ticker: str) -> csp.Outputs(average_spread=csp.ts[float], book_ts=ts[Book]):
    book_ts = book(ticker)
    return csp.output(average_spread=average_spread(book_ts), book_ts=book_ts)


@csp.node
def output_book_times_aux(book: ts[Book]) -> csp.ts[Book]:
    with csp.alarms():
        a_start = csp.alarm(bool)
        a_end = csp.alarm(bool)

    with csp.start():
        csp.schedule_alarm(a_start, timedelta(), True)
        csp.schedule_alarm(a_end, csp.engine_end_time(), True)

    if csp.ticked(book):
        if csp.ticked(a_end):
            csp.schedule_alarm(a_end, timedelta(), True)
        return book
    elif csp.ticked(a_start) or csp.ticked(a_end):
        return Book(exchange_timestamp=csp.now())


# This is a special graph, it's for advanced usage. Here we override the timestamp with which the data is written to cache.
# It's written using the exchange_timestamp from book_ts.exchange_timestamp. This kind of graphs must always be written in a separate
# engine run and consumed using get_book_with_average_spread_exchange_timestamp.cached
@csp.graph(cache=True, cache_options=GraphCacheOptions(data_timestamp_column_name="book_ts.exchange_timestamp"))
def get_book_with_average_spread_exchange_timestamp(ticker: str) -> csp.Outputs(
    average_spread=csp.ts[float], book_ts=ts[Book]
):
    return csp.output(
        average_spread=get_book_with_average_spread(ticker).average_spread,
        book_ts=output_book_times_aux(get_book_with_average_spread(ticker).book_ts),
    )


@csp.graph(cache=True, cache_options=GraphCacheOptions(category=["forecasts", "researched", "mean_reversion_based"]))
def dummy_mean_reversion_forecast(ticker: str) -> csp.Outputs(dummy_fc=csp.ts[float]):
    @csp.node
    def _calc(mid_price: csp.ts[float]) -> csp.ts[float]:
        """Compute a dummy mean reversion forecast
        :param sampled_prices:
        :return:
        """
        with csp.state():
            look_back_timedelta = timedelta(minutes=5)

        with csp.start():
            csp.set_buffering_policy(mid_price, tick_history=look_back_timedelta)

        prev_price = csp.value_at(mid_price, -look_back_timedelta, default=math.nan)

        if not math.isnan(prev_price):
            return prev_price - mid_price

    return csp.output(dummy_fc=_calc(mid(ticker)))


@csp.graph
def main_graph():
    csp.print("AAPL spread", get_book_with_average_spread("AAPL").average_spread)
    get_book_with_average_spread("IBM")
    csp.print("IBM spread", get_book_with_average_spread("IBM").average_spread)
    # The following line would fail, if uncommented.
    # graphs with custom data_timestamp_column_name must be first written and then consumed using .cached property
    # csp.print('IBM spread', get_book_with_average_spread_exchange_timestamp('AAPL').average_spread)
    # We can write them though without consuming
    get_book_with_average_spread_exchange_timestamp("AAPL")
    get_book_with_average_spread_exchange_timestamp("IBM")
    dummy_mean_reversion_forecast("AAPL")
    dummy_mean_reversion_forecast("IBM")


@csp.graph
def main_graph_cached():
    csp.print("AAPL spread", get_book_with_average_spread.cached("AAPL").average_spread)
    csp.print(
        "AAPL spread exchange timestamp", get_book_with_average_spread_exchange_timestamp.cached("AAPL").average_spread
    )
    # We can also read some specific columns from the struct, only bid will be read from file in this case:
    csp.print("AAPL bid", get_book_with_average_spread.cached("AAPL").book_ts.bid)


if __name__ == "__main__":
    output_folder = os.path.expanduser("~/csp_example_cache")
    output_folder_forecasts = os.path.join(output_folder, "overridden_forecast_output_folder")
    starttime = datetime(2020, 1, 1, 9, 29)
    endtime = starttime + timedelta(minutes=20)
    cache_config = CacheConfig(
        data_folder=output_folder,
        category_overrides=[
            CacheCategoryConfig(category=["forecasts", "researched"], data_folder=output_folder_forecasts)
        ],
    )
    csp.run(main_graph, starttime=starttime, endtime=endtime, config=Config(cache_config=cache_config))
    print("-" * 30 + " FINISHED MAIN GRAPH RUN" + "-" * 30)
    # After the cache was generated, we can run the graph that accesses cached only
    csp.run(
        main_graph_cached,
        starttime=starttime,
        endtime=endtime,
        config=Config(cache_config=CacheConfig(data_folder=output_folder)),
    )
    print("-" * 30 + " FINISHED CACHED GRAPH RUN" + "-" * 30)

    # We can now look into the dataset on disk:
    cached_data = get_book_with_average_spread.using().cached_data(output_folder)("AAPL")
    print("Files are:", cached_data.get_data_files_for_period(starttime, endtime))
    # We can also load the data as dataframe:
    print("AAPL data:\n", cached_data.get_data_df_for_period(starttime, endtime))
    # We can alternative load just some columns
    print(
        "AAPL data few columns:\n",
        cached_data.get_data_df_for_period(starttime, endtime, column_list=["book_ts.ask", "book_ts.bid"]),
    )
    # The following will raise an exception since we don't have the data for the given period in cache:
    try:
        print(
            "AAPL data few columns larger period:\n",
            cached_data.get_data_df_for_period(
                starttime - timedelta(seconds=1), endtime, column_list=["book_ts.ask", "book_ts.bid"]
            ),
        )
    except RuntimeError as e:
        print(f"Missing data error: {str(e)}")
    # The following will work, since we provide missing_range_handler that ignores the missing ranges
    print(
        "AAPL data few columns larger period:\n",
        cached_data.get_data_df_for_period(
            starttime - timedelta(seconds=1),
            endtime,
            missing_range_handler=lambda start, end: True,
            column_list=["book_ts.ask", "book_ts.bid"],
        ),
    )

    fc_cached_data = dummy_mean_reversion_forecast.cached_data(output_folder_forecasts)("AAPL")

    # alternative more convenient way to get cached data if cache config available is to use CacheDataPathResolver:
    data_path_resolver = CacheConfigResolver(cache_config)
    fc_cached_data2 = dummy_mean_reversion_forecast.cached_data(data_path_resolver)("AAPL")

    print(
        f"AAPL mean reversion forecast files:{list(fc_cached_data.get_data_files_for_period(starttime, endtime).values())}"
    )
    print(f"AAPL mean reversion forecast:\n{fc_cached_data.get_data_df_for_period(starttime, endtime)}")
    fc_cached_data.get_data_df_for_period(starttime, endtime) == fc_cached_data2.get_data_df_for_period(
        starttime, endtime
    )
    # The loaded data from both ways of accessing the cache should be the same
    assert (
        (
            fc_cached_data.get_data_df_for_period(starttime, endtime)
            == fc_cached_data2.get_data_df_for_period(starttime, endtime)
        )
        .all()
        .all()
    )

    # we can also read the forecasts directly as parquet table:
    appl_fc_arrow_table = fc_cached_data.get_data_df_for_period(
        starttime,
        endtime,
        data_loader_function=lambda starttime,
        endtime,
        data_file,
        column_list,
        basket_column_list: pyarrow.parquet.read_table(data_file, column_list),
    )[0]
    print(f"AAPL forecast using pyarrow read: {appl_fc_arrow_table['dummy_fc']}")
