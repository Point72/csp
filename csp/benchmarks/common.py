from asv_runner.benchmarks import benchmark_types
from asv_runner.benchmarks.mark import SkipNotImplemented
from logging import getLogger

__all__ = ("ASVBenchmarkHelper",)


class ASVBenchmarkHelper:
    """A helper base class to mimic some of what ASV does when running benchmarks, to
    test them outside of ASV.

    NOTE: should be removed in favor of calling ASV itself from python, if possible.
    """

    def __init__(self, *args, **kwargs):
        self.log = getLogger(self.__class__.__name__)

    def run_all(self):
        # https://asv.readthedocs.io/en/v0.6.3/writing_benchmarks.html#benchmark-types
        benchmarks = {}

        for method in dir(self):
            for cls in benchmark_types:
                if cls.name_regex.match(method):
                    benchmark_type = cls.__name__.replace("Benchmark", "")
                    if benchmark_type not in benchmarks:
                        benchmarks[benchmark_type] = []

                    name = f"{self.__class__.__qualname__}.{method}"
                    func = getattr(self, method)
                    benchmarks[benchmark_type].append(cls(name, func, (func, self)))

        def run_benchmark(benchmark):
            skip = benchmark.do_setup()
            try:
                if skip:
                    return
                try:
                    benchmark.do_run()
                except SkipNotImplemented:
                    pass
            finally:
                benchmark.do_teardown()

        for type, benchmarks_to_run in benchmarks.items():
            if benchmarks_to_run:
                self.log.warn(f"Running benchmarks for {type}")
            for benchmark in benchmarks_to_run:
                if len(getattr(self, "params", [])):
                    # TODO: cleaner
                    param_count = 0
                    while param_count < 100:
                        try:
                            benchmark.set_param_idx(param_count)
                            params = benchmark._current_params
                            self.log.warn(f"[{type}][{benchmark.name}][{'.'.join(str(_) for _ in params)}]")
                            run_benchmark(benchmark=benchmark)
                            param_count += 1
                        except ValueError:
                            break
                else:
                    self.log.warn(f"Running [{type}][{benchmark.func.__name__}]")
                    run_benchmark(benchmark=benchmark)
