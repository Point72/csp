# Written by @AdamGlustein
import numpy as np
import time
from datetime import datetime, timedelta

import csp

st = datetime(2020, 1, 1)
N = 10_000
ARRAY_SIZE = 100
TEST_TIMES = [st + timedelta(seconds=i) for i in range(N)]
RANDOM_VALUES = [np.random.normal(size=(ARRAY_SIZE,)) for i in range(N)]  # 100 element np array
DATA = list(zip(TEST_TIMES, RANDOM_VALUES))
INTERVAL = 1000
NUM_SAMPLES = 100


def g_qtl():
    data = csp.curve(typ=np.ndarray, data=DATA)
    median = csp.stats.median(data, interval=INTERVAL)
    csp.add_graph_output("final_median", median, tick_count=1)


def g_rank():
    data = csp.curve(typ=np.ndarray, data=DATA)
    rank = csp.stats.rank(data, interval=INTERVAL)
    csp.add_graph_output("final_rank", rank, tick_count=1)


if __name__ == "__main__":
    qtl_times, rank_times = list(), list()
    for i in range(NUM_SAMPLES):
        start = time.time()
        csp.run(g_qtl, starttime=st, endtime=timedelta(seconds=N))
        post_qtl = time.time()
        csp.run(g_rank, starttime=st, endtime=timedelta(seconds=N))
        post_rank = time.time()

        qtl_times.append(post_qtl - start)
        rank_times.append(post_rank - post_qtl)
        print(i)

    avg_med = sum(qtl_times) / NUM_SAMPLES
    avg_rank = sum(rank_times) / NUM_SAMPLES
    print(
        f"Average time in {NUM_SAMPLES} tests for median with {N=}, {ARRAY_SIZE=}, {INTERVAL=}: {round(avg_med, 2)} s"
    )
    print(f"Average time in {NUM_SAMPLES} tests for rank with {N=}, {ARRAY_SIZE=}, {INTERVAL=}: {round(avg_rank, 2)} s")
