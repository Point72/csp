import numpy as np
from datetime import datetime, timedelta
from timeit import Timer

import csp
from csp.benchmarks import ASVBenchmarkHelper

__all__ = ("StatsBenchmarkSuite",)


class StatsBenchmarkSuite(ASVBenchmarkHelper):
    """
    python -m csp.benchmarks.stats.basic
    """

    params = (("median", "quantile", "rank"),)
    param_names = ("function",)

    rounds = 5
    repeat = (100, 200, 60.0)

    function_args = {"quantile": {"quant": 0.95}}

    def setup(self, _):
        self.start_date = datetime(2020, 1, 1)
        self.num_rows = 1_000
        self.array_size = 100
        self.test_times = [self.start_date + timedelta(seconds=i) for i in range(self.num_rows)]
        self.random_values = [
            np.random.normal(size=(self.array_size,)) for i in range(self.num_rows)
        ]  # 100 element np array
        self.data = list(zip(self.test_times, self.random_values))
        self.interval = 500

    def time_stats(self, function):
        def g():
            data = csp.curve(typ=np.ndarray, data=self.data)
            value = getattr(csp.stats, function)(data, interval=self.interval, **self.function_args.get(function, {}))
            csp.add_graph_output("final_value", value, tick_count=1)

        timer = Timer(
            lambda: csp.run(g, realtime=False, starttime=self.start_date, endtime=timedelta(seconds=self.num_rows))
        )
        elapsed = timer.timeit(1)
        return elapsed


if __name__ == "__main__":
    sbs = StatsBenchmarkSuite()
    sbs.run_all()
