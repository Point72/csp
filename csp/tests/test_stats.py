import math
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np
import numpy.testing
import pandas as pd
from pandas.testing import assert_series_equal

import csp
from csp.stats import _window_updates
from csp.typing import Numpy1DArray


def list_nparr_to_matrix(x):
    return np.stack(np.array(x, dtype=object)[:, 1])


def aae(exp, act):
    np.testing.assert_almost_equal(np.array(exp, dtype=object)[:, 1], np.array(act, dtype=object)[:, 1], decimal=7)


def generate_random_data(n, mu, sigma, pnan):
    orig_state = np.random.get_state()
    np.random.seed(42)  # for reproducibility
    times = np.empty(n, dtype=object)
    values = np.random.normal(mu, sigma, size=n)
    deltas = np.random.uniform(low=0.0, high=10.0, size=n)

    times[0] = datetime(2020, 1, 1)
    for i in range(1, n):
        times[i] = times[i - 1] + timedelta(seconds=deltas[i])
        if np.random.random() > (1 - pnan):
            values[i] = float("nan")
    np.random.set_state(orig_state)

    if pnan:
        values[0] = float("nan")  # force edge condition
    return times, values


class TestStats(unittest.TestCase):
    def test_deque(self):
        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(datetime(2020, 1, i + 1), i + 1) for i in range(10)])
            y = csp.curve(
                typ=float,
                data=[
                    (datetime(2020, 1, 1), 1),
                    (datetime(2020, 1, 2), 2),
                    (datetime(2020, 1, 3), 3),
                    (datetime(2020, 1, 8), 8),
                    (datetime(2020, 1, 9), 9),
                    (datetime(2020, 1, 10), 10),
                ],
            )
            y_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (datetime(2020, 1, 1), np.array([1.0])),
                    (datetime(2020, 1, 2), np.array([2.0])),
                    (datetime(2020, 1, 3), np.array([3.0])),
                    (datetime(2020, 1, 8), np.array([8.0])),
                    (datetime(2020, 1, 9), np.array([9.0])),
                    (datetime(2020, 1, 10), np.array([10.0])),
                ],
            )
            trigger = csp.curve(typ=float, data=[(datetime(2020, 1, 4), True), (datetime(2020, 1, 12), True)])
            recalc = csp.curve(
                typ=float,
                data=[(datetime(2020, 1, 1), True), (datetime(2020, 1, 4), True), (datetime(2020, 1, 7), True)],
            )

            basic_dq_time = _window_updates(x, timedelta(days=3), x, x, csp.null_ts(bool), csp.null_ts(bool))
            basic_dq_tick = _window_updates(x, 3, x, x, csp.null_ts(bool), csp.null_ts(bool))

            trig_dq_time = _window_updates(y, timedelta(days=5), trigger, y, csp.null_ts(bool), csp.null_ts(bool))
            trig_dq_tick = _window_updates(y, 3, trigger, y, csp.null_ts(bool), csp.null_ts(bool))

            trig_dq_np_time = _window_updates(
                y_np, timedelta(days=5), trigger, y_np, csp.null_ts(bool), csp.null_ts(bool)
            )
            trig_dq_np_tick = _window_updates(
                y_np, timedelta(days=5), trigger, y_np, csp.null_ts(bool), csp.null_ts(bool)
            )

            recalc_dq_tick = _window_updates(x, timedelta(days=5), x, x, csp.null_ts(bool), recalc)
            recalc_dq_time = _window_updates(x, timedelta(days=5), x, x, csp.null_ts(bool), recalc)

            csp.add_graph_output("remove_t", basic_dq_time.removals)
            csp.add_graph_output("add_t", basic_dq_time.additions)
            csp.add_graph_output("remove_n", basic_dq_tick.removals)
            csp.add_graph_output("add_n", basic_dq_tick.additions)

            csp.add_graph_output("order_rem_t", trig_dq_time.removals)
            csp.add_graph_output("order_add_t", trig_dq_time.additions)
            csp.add_graph_output("order_rem_n", trig_dq_tick.removals)
            csp.add_graph_output("order_add_n", trig_dq_tick.additions)

            csp.add_graph_output("np_remove_t", trig_dq_np_time.removals)
            csp.add_graph_output("np_add_t", trig_dq_np_time.additions)
            csp.add_graph_output("np_remove_n", trig_dq_np_tick.removals)
            csp.add_graph_output("np_add_n", trig_dq_np_tick.additions)

            csp.add_graph_output("recalc_rem_t", recalc_dq_time.removals)
            csp.add_graph_output("recalc_add_t", recalc_dq_time.additions)
            csp.add_graph_output("recalc_rem_n", recalc_dq_tick.removals)
            csp.add_graph_output("recalc_add_n", recalc_dq_tick.additions)

        st = datetime(2020, 1, 1)
        expected_rem = [(datetime(2020, 1, i), [i - 3]) for i in range(4, 11)]
        expected_add = [(datetime(2020, 1, i + 1), [i + 1]) for i in range(10)]
        results = csp.run(graph, starttime=st, endtime=st + timedelta(days=12))
        self.assertEqual(expected_rem, results["remove_t"])
        self.assertEqual(expected_add, results["add_t"])
        self.assertEqual(expected_rem, results["remove_n"])
        self.assertEqual(expected_add, results["add_n"])

        # note that for NumPy, we always push the first tick as an addition whether or not it was triggered. We do this for float too now
        # this is so the computation objects know the shape of the array so that, if triggered, they can return an all-NaN array with the proper shape

        # test order with multiple additions and removals
        expected_rem = [(datetime(2020, 1, 12), [1, 2, 3])]
        expected_add = [
            (datetime(2020, 1, 1), [1]),
            (datetime(2020, 1, 4), [2, 3]),
            (datetime(2020, 1, 12), [8, 9, 10]),
        ]
        expected_add_np = [
            (datetime(2020, 1, 1), [np.array([1.0])]),
            (datetime(2020, 1, 4), [np.array([2.0]), np.array([3.0])]),
            (datetime(2020, 1, 12), [np.array([i + 8], dtype=float) for i in range(3)]),
        ]
        expected_rem_np = [(datetime(2020, 1, 12), [np.array([i + 1], dtype=float) for i in range(3)])]

        self.assertEqual(expected_rem, results["order_rem_t"])
        self.assertEqual(expected_add, results["order_add_t"])
        self.assertEqual(expected_rem, results["order_rem_n"])
        self.assertEqual(expected_add, results["order_add_n"])

        self.assertEqual(expected_rem_np, results["np_remove_t"])
        self.assertEqual(expected_add_np, results["np_add_t"])
        self.assertEqual(expected_rem_np, results["np_remove_n"])
        self.assertEqual(expected_add_np, results["np_add_n"])

        # recalc handling
        expected_rem = [
            (datetime(2020, 1, 1) + timedelta(days=i), [i - 4]) for i in [5, 7, 8, 9]
        ]  # no removal on recalc day 7
        expected_add = [(datetime(2020, 1, 1) + timedelta(days=i), [i + 1]) for i in range(10)]
        # add in recalc at days 4 and 7
        expected_add[3] = (datetime(2020, 1, 4), [1, 2, 3, 4])
        expected_add[6] = (datetime(2020, 1, 7), [3, 4, 5, 6, 7])

        self.assertEqual(expected_rem, results["recalc_rem_t"])
        self.assertEqual(expected_add, results["recalc_add_t"])
        self.assertEqual(expected_rem, results["recalc_rem_n"])
        self.assertEqual(expected_add, results["recalc_add_n"])

    def test_time_window_update_bug(self):
        '''test for bug in "csp.stats time window bug on duplicate timestamps"'''

        def g():
            data = csp.curve(
                float,
                [
                    (datetime(2023, 2, 17, 0), 1.0),
                    (datetime(2023, 2, 17, 1), 2.0),
                    (datetime(2023, 2, 17, 1), 3.0),
                    (datetime(2023, 2, 17, 1), 4.0),
                    (datetime(2023, 2, 17, 5), 5.0),
                ],
            )

            data_np = csp.curve(
                np.ndarray,
                [
                    (datetime(2023, 2, 17, 0), np.array([1.0, 1.0])),
                    (datetime(2023, 2, 17, 1), np.array([2.0, 2.0])),
                    (datetime(2023, 2, 17, 1), np.array([3.0, 3.0])),
                    (datetime(2023, 2, 17, 1), np.array([4.0, 4.0])),
                    (datetime(2023, 2, 17, 5), np.array([5.0, 5.0])),
                ],
            )

            trigger = csp.curve(
                bool,
                [
                    (datetime(2023, 2, 17, 1), True),
                    (datetime(2023, 2, 17, 6), True),
                    (datetime(2023, 2, 17, 10), True),
                ],
            )

            updates_f = _window_updates(data, timedelta(hours=3), trigger, data, csp.null_ts(bool), csp.null_ts(bool))
            updates_np = _window_updates(
                data_np, timedelta(hours=3), trigger, data_np, csp.null_ts(bool), csp.null_ts(bool)
            )
            csp.add_graph_output("ADD_F", updates_f.additions)
            csp.add_graph_output("REM_F", updates_f.removals)
            csp.add_graph_output("ADD_NP", updates_np.additions)
            csp.add_graph_output("REM_NP", updates_np.removals)

        res = csp.run(g, starttime=datetime(2023, 2, 17))
        all_adds_f = [v for t in res["ADD_F"] for v in t[1]]
        all_rem_f = [v for t in res["REM_F"] for v in t[1]]
        all_adds_np = [v for t in res["ADD_NP"] for v in t[1]]
        all_rem_np = [v for t in res["REM_NP"] for v in t[1]]

        self.assertEqual(all_adds_f, all_rem_f)
        np.testing.assert_equal(all_adds_np, all_rem_np)

    def test_rolling_sum_mean(self):
        dvalues = np.random.randint(low=-100, high=100, size=(100,)).astype(np.float64)
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), int(dvalues[i])) for i in range(100)])
            x_i = csp.curve(typ=int, data=[(st + timedelta(seconds=i + 1), int(dvalues[i])) for i in range(100)])

            # By time interval
            rcount_t = csp.stats.count(x, interval=timedelta(seconds=5), min_window=timedelta(seconds=3))
            rsums_t = csp.stats.sum(x, interval=timedelta(seconds=5), min_window=timedelta(seconds=3))
            rsums_t_int = csp.stats.sum(x_i, interval=timedelta(seconds=5), min_window=timedelta(seconds=3))
            rmeans_t = csp.stats.mean(x, interval=timedelta(seconds=5))
            # By tick interval
            rcount_n = csp.stats.count(x, interval=5, min_window=3)
            rsums_n = csp.stats.sum(x, interval=5, min_window=3, precise=True)
            rmeans_n = csp.stats.mean(x, interval=5)
            rmeans_n_int = csp.stats.mean(x_i, interval=5)

            csp.add_graph_output("count_t", rcount_t)
            csp.add_graph_output("sums_t", rsums_t)
            csp.add_graph_output("means_t", rmeans_t)
            csp.add_graph_output("count_n", rcount_n)
            csp.add_graph_output("sums_n", rsums_n)
            csp.add_graph_output("means_n", rmeans_n)
            csp.add_graph_output("sums_t_int", rsums_t_int)
            csp.add_graph_output("means_n_int", rmeans_n_int)

        values = pd.Series(dvalues)

        rcount = values.rolling(window=5, min_periods=3).count().tolist()
        rsums = values.rolling(window=5, min_periods=3).sum().tolist()
        rmeans = values.rolling(window=5, min_periods=3).mean().tolist()

        expected_count = [(st + timedelta(seconds=i), rcount[i - 1]) for i in range(3, 101)]
        expected_sums = [(st + timedelta(seconds=i), rsums[i - 1]) for i in range(3, 101)]
        expected_means = np.array([rmeans[i - 1] for i in range(5, 101)])
        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        # tests (t for time, n for tick)
        self.assertEqual(expected_count, results["count_t"])
        self.assertEqual(expected_sums, results["sums_t"])
        self.assertEqual(expected_sums, results["sums_t_int"])
        np.testing.assert_almost_equal(expected_means, np.array(results["means_t"])[:, 1], decimal=7)
        self.assertEqual(expected_count, results["count_n"])
        self.assertEqual(expected_sums, results["sums_n"])
        np.testing.assert_almost_equal(expected_means, np.array(results["means_n"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_means, np.array(results["means_n_int"])[:, 1], decimal=7)

    def test_unique(self):
        dvalues = [1, 0.3, (0.1 + 0.2), float("nan"), float("nan"), -4, 0, 5, float("nan"), 9]
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), dvalues[i]) for i in range(10)])

            trigger = csp.curve(typ=float, data=[(st + timedelta(seconds=7), True), (st + timedelta(seconds=9), True)])

            unique_t = csp.stats.unique(x, timedelta(seconds=5), timedelta(seconds=3))
            unique_n = csp.stats.unique(x, 5, 3)

            csp.add_graph_output("unique_t", unique_t)
            csp.add_graph_output("unique_n", unique_n)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

        exp_unique = [2, 2, 2, 2, 3, 3, 3, 4]
        exp_unique = [(st + timedelta(seconds=i + 3), exp_unique[i]) for i in range(8)]

        self.assertEqual(exp_unique, results["unique_t"])
        self.assertEqual(exp_unique, results["unique_n"])

    def test_first_last_product(self):
        dvalues = [1, 2, 2, float("nan"), float("nan"), -4, 0, 5, float("nan"), 9]
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), dvalues[i]) for i in range(10)])

            trigger = csp.curve(typ=float, data=[(st + timedelta(seconds=7), True), (st + timedelta(seconds=9), True)])

            first_t = csp.stats.first(x, interval=timedelta(seconds=3), min_window=timedelta(seconds=3))
            first_n = csp.stats.first(x, 3, 3)
            last_t = csp.stats.last(x, timedelta(seconds=5), timedelta(seconds=3), True)
            last_n = csp.stats.last(x, 5, 3, True)
            prod_t = csp.stats.prod(x, timedelta(seconds=5), timedelta(seconds=3))
            prod_n = csp.stats.prod(x, 5, 3)

            csp.add_graph_output("first_t", first_t)
            csp.add_graph_output("first_n", first_n)
            csp.add_graph_output("last_t", last_t)
            csp.add_graph_output("last_n", last_n)
            csp.add_graph_output("prod_t", prod_t)
            csp.add_graph_output("prod_n", prod_n)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

        exp_first = [1, 2, 2, -4, -4, -4, 0, 5]
        exp_first = [(st + timedelta(seconds=i + 3), exp_first[i]) for i in range(8)]
        exp_prod = [4, 4, 4, -16, 0, 0, 0, 0]
        exp_prod = [(st + timedelta(seconds=i + 3), exp_prod[i]) for i in range(8)]
        exp_last = [2, 2, 2, -4, 0, 5, 5, 9]
        exp_last = [(st + timedelta(seconds=i + 3), exp_last[i]) for i in range(8)]

        self.assertEqual(exp_first, results["first_t"])
        self.assertEqual(exp_first, results["first_n"])
        self.assertEqual(exp_last, results["last_t"])
        self.assertEqual(exp_last, results["last_n"])
        self.assertEqual(exp_prod, results["prod_t"])
        self.assertEqual(exp_prod, results["prod_n"])

    def test_first_last_ignore_na_false(self):
        st = datetime(2020, 1, 1, 0, 0)
        out = csp.stats.last(csp.unroll(csp.const([1.0, np.nan, 3.0, 4.0])), 2, ignore_na=False, min_window=1).run(
            starttime=st
        )
        exp = [(st, 1.0), (st, np.nan), (st, 3.0), (st, 4.0)]
        self.assertEqual(len(exp), len(out))
        # we do this for the nan checks
        for e, o in zip(exp, out):
            np.testing.assert_equal(e, o)

        dvalues = [1, 2, 2, float("nan"), float("nan"), -4, 0, 5, float("nan"), 9]
        x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), v) for i, v in enumerate(dvalues)])
        out2 = csp.stats.last(x, timedelta(seconds=5), ignore_na=False).run(starttime=st)
        exp2 = [
            (st + timedelta(seconds=5), np.nan),
            (st + timedelta(seconds=6), -4),
            (st + timedelta(seconds=7), 0.0),
            (st + timedelta(seconds=8), 5.0),
            (st + timedelta(seconds=9), np.nan),
            (st + timedelta(seconds=10), 9.0),
        ]
        self.assertEqual(len(exp2), len(out2))
        for e, o in zip(exp2, out2):
            np.testing.assert_equal(e, o)

        out3 = csp.stats.first(csp.unroll(csp.const([1.0, np.nan, 3.0, 4.0])), 4, ignore_na=False, min_window=1).run(
            starttime=st
        )
        self.assertEqual(out3, [(st, 1.0)] * 4)

        out4 = csp.stats.first(csp.unroll(csp.const([1.0, np.nan, 3.0, 4.0])), 2, ignore_na=False, min_window=1).run(
            starttime=st
        )
        exp4 = [(st, 1.0), (st, 1.0), (st, np.nan), (st, 3.0)]
        self.assertEqual(len(exp4), len(out4))
        for e, o in zip(exp4, out4):
            np.testing.assert_equal(e, o)

        out5 = csp.stats.first(csp.unroll(csp.const([1.0, np.nan, 3.0, 4.0])), 2, ignore_na=False, min_window=2).run(
            starttime=st
        )
        exp5 = [(st, 1.0), (st, np.nan), (st, 3.0)]
        self.assertEqual(len(exp5), len(out5))
        for e, o in zip(exp5, out5):
            np.testing.assert_equal(e, o)

    def test_rolling_min_max(self):
        dvalues = np.random.uniform(low=-100, high=100, size=(30,))
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(
                typ=float, data=[(datetime(2020, 1, 1) + timedelta(milliseconds=i + 1), dvalues[i]) for i in range(30)]
            )
            # By time interval
            rmax_t = csp.stats.max(x, interval=timedelta(milliseconds=5), min_window=timedelta(milliseconds=3))
            rmin_t = csp.stats.min(x, interval=timedelta(milliseconds=5), min_window=timedelta(milliseconds=5))
            # By tick interval
            rmax_n = csp.stats.max(x, interval=5, min_window=3)
            rmin_n = csp.stats.min(x, interval=5)

            csp.add_graph_output("max_t", rmax_t)
            csp.add_graph_output("min_t", rmin_t)
            csp.add_graph_output("max_n", rmax_n)
            csp.add_graph_output("min_n", rmin_n)

        values = pd.Series(dvalues)
        rmax = values.rolling(window=5, min_periods=3).max().tolist()
        rmin = values.rolling(window=5, min_periods=3).min().tolist()

        expected_max = [rmax[i - 1] for i in range(3, 31)]
        expected_min = [rmin[i - 1] for i in range(5, 31)]

        results = csp.run(graph, starttime=st, endtime=st + timedelta(milliseconds=30))

        # tests (t for time, n for tick)
        np.testing.assert_almost_equal(expected_max, np.array(results["max_t"], dtype=object)[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_max, np.array(results["max_n"], dtype=object)[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_min, np.array(results["min_t"], dtype=object)[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_min, np.array(results["min_n"], dtype=object)[:, 1], decimal=7)

    def test_rolling_var_sd_sem_skew_kurt(self):
        dvalues = np.random.uniform(low=-1000, high=1000, size=(1000,))
        ddoft = 1
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), dvalues[i]) for i in range(1000)])
            var_t = csp.stats.var(x, timedelta(seconds=500), timedelta(seconds=100), ddoft)
            var_n = csp.stats.var(x, 500, 100, ddoft)
            sd_t = csp.stats.stddev(x, timedelta(seconds=500), ddof=ddoft)
            sd_n = csp.stats.stddev(x, 500, ddof=ddoft)

            sem_t = csp.stats.sem(x, timedelta(seconds=500), timedelta(seconds=2), ddof=ddoft)
            sem_n = csp.stats.sem(x, 500, 2, ddof=ddoft)
            skew_t_unbias = csp.stats.skew(x, timedelta(seconds=500), timedelta(seconds=2))
            skew_n_bias = csp.stats.skew(x, 500, 2, bias=True)
            kurt_t_excess = csp.stats.kurt(x, timedelta(seconds=500), timedelta(seconds=2))
            kurt_n = csp.stats.kurt(x, 500, 2, bias=False, excess=False)

            csp.add_graph_output("var_t", var_t)
            csp.add_graph_output("var_n", var_n)
            csp.add_graph_output("sd_t", sd_t)
            csp.add_graph_output("sd_n", sd_n)

            csp.add_graph_output("sem_t", sem_t)
            csp.add_graph_output("sem_n", sem_n)
            csp.add_graph_output("skew_t_unbias", skew_t_unbias)
            csp.add_graph_output("skew_n_bias", skew_n_bias)
            csp.add_graph_output("kurt_t_excess", kurt_t_excess)
            csp.add_graph_output("kurt_n", kurt_n)

        values = pd.Series(dvalues)
        pdvar = values.rolling(window=500, min_periods=100).var(ddof=ddoft).to_numpy()
        pdsd = values.rolling(window=500, min_periods=499).std(ddof=ddoft).to_numpy()
        expected_var = np.array([(st + timedelta(seconds=i + 1), pdvar[i]) for i in range(1000)])
        expected_sd = np.array([(st + timedelta(seconds=i + 1), pdsd[i]) for i in range(1000)])
        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=1000))

        # tests (t for time, n for tick)
        # check datetimes
        np.testing.assert_equal(expected_var[99:, 0], np.array(results["var_t"])[:, 0])
        np.testing.assert_equal(expected_sd[499:, 0], np.array(results["sd_n"])[:, 0])

        # floats, ensure accurate to 1e-6
        np.testing.assert_almost_equal(expected_var[99:, 1], np.array(results["var_t"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_var[99:, 1], np.array(results["var_n"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_sd[499:, 1], np.array(results["sd_t"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_sd[499:, 1], np.array(results["sd_n"])[:, 1], decimal=7)

        pdsem = values.rolling(window=500, min_periods=2).sem(ddof=ddoft).to_numpy()
        pdskew_unbias = values.rolling(window=500, min_periods=2).skew().to_numpy()
        pdkurt_unbias = values.rolling(window=500, min_periods=2).kurt().to_numpy()

        expected_sem = np.array([(st + timedelta(seconds=i + 1), pdsem[i]) for i in range(1000)])
        expected_skew_unbias = np.array([(st + timedelta(seconds=i + 1), pdskew_unbias[i]) for i in range(1000)])
        expected_kurt_excess = np.array([(st + timedelta(seconds=i + 1), pdkurt_unbias[i]) for i in range(1000)])

        np.testing.assert_almost_equal(expected_sem[1:, 1], np.array(results["sem_t"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(expected_sem[1:, 1], np.array(results["sem_n"])[:, 1], decimal=7)

        unbiased_skew = np.array(results["skew_t_unbias"])[:, 1].astype(float)
        biased_skew = np.array(results["skew_n_bias"])[:, 1].astype(float)
        excess_kurtosis = np.array(results["kurt_t_excess"])[:, 1].astype(float)
        kurtosis = np.array(results["kurt_n"])[:, 1].astype(float)
        np.testing.assert_almost_equal(expected_skew_unbias[1:, 1].astype(float), unbiased_skew, decimal=7)
        np.testing.assert_almost_equal(expected_kurt_excess[1:, 1].astype(float), excess_kurtosis, decimal=7)

        # only scipy has unbiased skew to test against, not worth the dependency
        # instead, just ensure parameters are being passed correctly by ensuring results are different
        np.testing.assert_equal(np.all(np.not_equal(unbiased_skew[:50], biased_skew[:50])), True)

        # can test for non-excess kurtosis easily (assuming our excess kurtosis is correct, of course...)
        np.testing.assert_almost_equal(excess_kurtosis, kurtosis - 3, decimal=7)

    def test_ema(self):
        N = 1000
        dvalues = np.random.uniform(low=-100, high=100, size=(N,))
        dvalues[0] = np.nan  # this forces edge cases around first value being nan
        for i in range(N):
            p = np.random.rand()
            if p < 0.2:
                dvalues[i] = np.nan

        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(milliseconds=i + 1), dvalues[i]) for i in range(N)])
            ema = csp.stats.ema(x, alpha=0.1, adjust=False)
            ema_var = csp.stats.ema_var(x, min_periods=3, span=20, adjust=True, bias=True)
            ema_std = csp.stats.ema_std(x, min_periods=3, span=20, adjust=True, bias=False)
            ema_std2 = csp.stats.ema_std(x, min_periods=3, span=20, adjust=False, ignore_na=False, bias=False)

            csp.add_graph_output("ema", ema)
            csp.add_graph_output("ema_v", ema_var)
            csp.add_graph_output("ema_s", ema_std)
            csp.add_graph_output("ema_s2", ema_std2)

        values = pd.Series(dvalues)
        pd_alpha = values.ewm(alpha=0.1, adjust=False).mean()
        pd_span = values.ewm(span=20, adjust=True)
        pd_var = pd_span.var(bias=True)
        pd_std = pd_span.std(bias=False)

        pd_span2 = values.ewm(span=20, adjust=False, ignore_na=False)
        pd_std2 = pd_span2.std(bias=False)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(milliseconds=1000))

        # floats, ensure accurate to 1.5e-7
        np.testing.assert_allclose(
            np.array(pd_alpha), np.array(results["ema"])[:, 1].astype(np.float64), atol=1.5e-7, equal_nan=True
        )
        np.testing.assert_allclose(
            np.array(pd_var)[2:], np.array(results["ema_v"])[:, 1].astype(np.float64), atol=1.5e-7, equal_nan=True
        )
        np.testing.assert_allclose(
            np.array(pd_std)[2:], np.array(results["ema_s"])[:, 1].astype(np.float64), atol=1.5e-7, equal_nan=True
        )
        np.testing.assert_allclose(
            np.array(pd_std2)[2:], np.array(results["ema_s2"])[:, 1].astype(np.float64), atol=1.5e-7, equal_nan=True
        )

    def test_triggers(self):
        dvalues = [i + 1 for i in range(20)]
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            trigger = csp.curve(
                typ=bool,
                data=[
                    (st + timedelta(seconds=3.5), True),
                    (st + timedelta(seconds=6.5), True),
                    (st + timedelta(seconds=8.5), True),
                    (st + timedelta(seconds=9.5), True),
                    (st + timedelta(seconds=16.5), True),
                    (st + timedelta(seconds=23.5), True),
                    (st + timedelta(seconds=26.5), True),
                ],
            )

            x = csp.curve(
                typ=int,
                data=[(st + timedelta(seconds=i + 1), dvalues[i]) for i in range(20)]
                + [(st + timedelta(seconds=25), 2), (st + timedelta(seconds=26), 2)],
            )

            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (
                        st + timedelta(seconds=i + 1),
                        np.array([[dvalues[i], dvalues[i]], [dvalues[i], dvalues[i]]], dtype=float),
                    )
                    for i in range(20)
                ]
                + [
                    (st + timedelta(seconds=25), np.array([[2.0, 2.0], [2.0, 2.0]])),
                    (st + timedelta(seconds=26), np.array([[2.0, 2.0], [2.0, 2.0]])),
                ],
            )

            # calculate values only on trigger
            dq = _window_updates(
                csp.cast_int_to_float(x),
                interval=timedelta(seconds=3),
                trigger=trigger,
                sampler=x,
                reset=csp.null_ts(float),
                recalc=csp.null_ts(float),
            )
            tsum = csp.stats.sum(x, interval=timedelta(seconds=3), trigger=trigger)
            tmean = csp.stats.mean(x, interval=timedelta(seconds=3), min_window=timedelta(seconds=4), trigger=trigger)
            tstd = csp.stats.stddev(x, interval=timedelta(seconds=3), min_window=timedelta(seconds=4), trigger=trigger)
            tmax = csp.stats.max(x, interval=timedelta(seconds=3), min_window=timedelta(seconds=4), trigger=trigger)
            tgmean = csp.stats.gmean(x, interval=timedelta(seconds=3), min_window=timedelta(seconds=4), trigger=trigger)

            nsum = csp.stats.sum(x, 3, trigger=trigger)
            nmean = csp.stats.mean(x, 3, 4, trigger=trigger)
            nstd = csp.stats.stddev(x, 3, 4, trigger=trigger)
            nmax = csp.stats.max(x, 3, 4, trigger=trigger)
            ngmean = csp.stats.gmean(x, 3, 4, trigger=trigger)

            csp.add_graph_output("add", dq.additions)
            csp.add_graph_output("rm", dq.removals)
            csp.add_graph_output("tsum", tsum)
            csp.add_graph_output("tmean", tmean)
            csp.add_graph_output("tstd", tstd)
            csp.add_graph_output("tmax", tmax)
            csp.add_graph_output("tgmean", tgmean)

            csp.add_graph_output("nsum", nsum)
            csp.add_graph_output("nmean", nmean)
            csp.add_graph_output("nstd", nstd)
            csp.add_graph_output("nmax", nmax)
            csp.add_graph_output("ngmean", ngmean)

            # NumPy test

            np_sum = csp.stats.sum(x_np, timedelta(seconds=3), trigger=trigger)
            np_mean = csp.stats.mean(x_np, timedelta(seconds=3), trigger=trigger)

            csp.add_graph_output("np_sum", np_sum)
            csp.add_graph_output("np_mean", np_mean)

        # By time
        # tests cover initialization, disappearing windows, and reappearing windows
        exp_sum = [
            (st + timedelta(seconds=3.5), 6),
            (st + timedelta(seconds=6.5), 15),
            (st + timedelta(seconds=8.5), 21),
            (st + timedelta(seconds=9.5), 24),
            (st + timedelta(seconds=16.5), 45),
            (st + timedelta(seconds=23.5), 0),
            (st + timedelta(seconds=26.5), 4),
        ]

        exp_mean = [
            (st + timedelta(seconds=6.5), 5),
            (st + timedelta(seconds=8.5), 7),
            (st + timedelta(seconds=9.5), 8),
            (st + timedelta(seconds=16.5), 15),
            (st + timedelta(seconds=23.5), float("nan")),
            (st + timedelta(seconds=26.5), 2),
        ]

        exp_std = [
            (st + timedelta(seconds=6.5), np.std([4, 5, 6], ddof=1)),
            (st + timedelta(seconds=8.5), np.std([6, 7, 8], ddof=1)),
            (st + timedelta(seconds=9.5), np.std([7, 8, 9], ddof=1)),
            (st + timedelta(seconds=16.5), np.std([14, 15, 16], ddof=1)),
            (st + timedelta(seconds=23.5), float("nan")),
            (st + timedelta(seconds=26.5), 0),
        ]

        exp_max = [
            (st + timedelta(seconds=6.5), 6),
            (st + timedelta(seconds=8.5), 8),
            (st + timedelta(seconds=9.5), 9),
            (st + timedelta(seconds=16.5), 16),
            (st + timedelta(seconds=23.5), float("nan")),
            (st + timedelta(seconds=26.5), 2),
        ]

        exp_gmean = [
            (st + timedelta(seconds=6.5), 120 ** (1 / 3)),
            (st + timedelta(seconds=8.5), 336 ** (1 / 3)),
            (st + timedelta(seconds=9.5), 504 ** (1 / 3)),
            (st + timedelta(seconds=16.5), 3360 ** (1 / 3)),
            (st + timedelta(seconds=23.5), float("nan")),
            (st + timedelta(seconds=26.5), 2),
        ]

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=1000))

        aae(exp_sum, results["tsum"])
        aae(exp_mean[:-2], results["tmean"][:-2])
        aae(exp_mean[:-2], results["tmean"][:-2])
        aae(exp_max[:-2], results["tmax"][:-2])

        # NumPy
        for i in range(len(exp_sum)):
            exps = np.zeros(shape=(2, 2))
            exps.fill(exp_sum[i][1])
            np.testing.assert_almost_equal(exps, results["np_sum"][i][1])

        # stddev
        aae(exp_std[:-2], results["tstd"][:-2])
        self.assertTrue(math.isnan(results["tstd"][-2][1]))
        aae([exp_std[-1]], [results["tstd"][-1]])

        self.assertTrue(math.isnan(results["tmean"][-2][1]))
        self.assertTrue(math.isnan(results["tmax"][-2][1]))
        self.assertEqual(exp_sum[-1], results["tsum"][-1])
        aae([exp_mean[-1]], [results["tmean"][-1]])
        self.assertEqual(exp_max[-1], results["tmax"][-1])

        aae(exp_gmean[:-2], results["tgmean"][:-2])
        self.assertTrue(math.isnan(results["tgmean"][-2][1]))
        aae([exp_gmean[-1]], [results["tgmean"][-1]])

        # By tick
        # need to readjust the tests a bit

        exp_sum[-2:] = [(st + timedelta(seconds=23.5), 57), (st + timedelta(seconds=26.5), 24)]

        exp_mean[-2:] = [(st + timedelta(seconds=23.5), 19), (st + timedelta(seconds=26.5), 8)]

        exp_std[-2:] = [
            (st + timedelta(seconds=23.5), np.std([18, 19, 20], ddof=1)),
            (st + timedelta(seconds=26.5), np.std([20, 2, 2], ddof=1)),
        ]

        exp_max[-2:] = [(st + timedelta(seconds=23.5), 20), (st + timedelta(seconds=26.5), 20)]

        exp_gmean[-2:] = [
            (st + timedelta(seconds=23.5), 6840 ** (1 / 3)),
            (st + timedelta(seconds=26.5), 80 ** (1 / 3)),
        ]

        aae(exp_sum, results["nsum"])
        aae(exp_mean, results["nmean"])
        aae(exp_max, results["nmax"])
        aae(exp_std, results["nstd"])
        aae(exp_gmean, results["ngmean"])

    def test_nan_handling_base(self):
        # N = 10
        my_curve = [1, 2, 3, float("nan"), 4, float("nan"), float("nan"), 5, 6, 7]
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            y = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), val) for i, val in enumerate(my_curve)])

            # Test nan handling with nans in the time-series
            # Count
            c_ig_n = csp.stats.count(y, 3, 1, True)
            c_wn_n = csp.stats.count(y, 3, 1, False)
            c_ig_t = csp.stats.count(y, timedelta(seconds=3), timedelta(seconds=1), True)
            c_wn_t = csp.stats.count(y, timedelta(seconds=3), timedelta(seconds=1), False)
            csp.add_graph_output("c_ig_n", c_ig_n)
            csp.add_graph_output("c_wn_n", c_wn_n)
            csp.add_graph_output("c_ig_t", c_ig_t)
            csp.add_graph_output("c_wn_t", c_wn_t)

            # Sum
            s_ig_n = csp.stats.sum(y, 3, 3, ignore_na=True)
            s_wn_n = csp.stats.sum(y, 3, 3, precise=True, ignore_na=False)
            s_ig_t = csp.stats.sum(y, timedelta(seconds=3), timedelta(seconds=3), precise=True, ignore_na=True)
            s_wn_t = csp.stats.sum(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False)
            csp.add_graph_output("s_ig_n", s_ig_n)
            csp.add_graph_output("s_wn_n", s_wn_n)
            csp.add_graph_output("s_ig_t", s_ig_t)
            csp.add_graph_output("s_wn_t", s_wn_t)

            # Max (no need to test min)
            mx_ig_n = csp.stats.max(y, 3, 3, ignore_na=True)
            mx_wn_n = csp.stats.max(y, 3, 3, ignore_na=False)
            mx_ig_t = csp.stats.max(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True)
            mx_wn_t = csp.stats.max(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False)
            csp.add_graph_output("mx_ig_n", mx_ig_n)
            csp.add_graph_output("mx_wn_n", mx_wn_n)
            csp.add_graph_output("mx_ig_t", mx_ig_t)
            csp.add_graph_output("mx_wn_t", mx_wn_t)

            # mean
            mu_ig_n = csp.stats.mean(y, 3, 3, ignore_na=True)
            mu_wn_n = csp.stats.mean(y, 3, 3, ignore_na=False)
            mu_ig_t = csp.stats.mean(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True)
            mu_wn_t = csp.stats.mean(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False)
            csp.add_graph_output("mu_ig_n", mu_ig_n)
            csp.add_graph_output("mu_wn_n", mu_wn_n)
            csp.add_graph_output("mu_ig_t", mu_ig_t)
            csp.add_graph_output("mu_wn_t", mu_wn_t)

            # stddev
            std_ig_n = csp.stats.stddev(y, 3, 3, ignore_na=True)
            std_wn_n = csp.stats.stddev(y, 3, 3, ignore_na=False)
            std_ig_t = csp.stats.stddev(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True)
            std_wn_t = csp.stats.stddev(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False)
            csp.add_graph_output("std_ig_n", std_ig_n)
            csp.add_graph_output("std_wn_n", std_wn_n)
            csp.add_graph_output("std_ig_t", std_ig_t)
            csp.add_graph_output("std_wn_t", std_wn_t)

            # quantile
            qt_ig_n = csp.stats.quantile(y, 3, [0.25, 0.5, 0.75], 3, "midpoint")
            qt_wn_n = csp.stats.quantile(y, 3, [0.25, 0.5, 0.75], 3, "midpoint", ignore_na=False)
            med_ig_n = csp.stats.median(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True)
            med_wn_n = csp.stats.median(y, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False)
            csp.add_graph_output("qt_ig_n0", qt_ig_n[0])
            csp.add_graph_output("qt_ig_n1", qt_ig_n[1])
            csp.add_graph_output("qt_ig_n2", qt_ig_n[2])
            csp.add_graph_output("qt_wn_n0", qt_wn_n[0])
            csp.add_graph_output("qt_wn_n1", qt_wn_n[1])
            csp.add_graph_output("qt_wn_n2", qt_wn_n[2])
            csp.add_graph_output("med_ig_n", med_ig_n)
            csp.add_graph_output("med_wn_n", med_wn_n)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

        # test base funcs - count
        expected_count_ig = np.array([1, 2, 3, 2, 2, 1, 1, 1, 2, 3])
        expected_count_wn = np.array(
            [1, 2, 3, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), 3]
        )
        np.testing.assert_equal(expected_count_ig, np.array(results["c_ig_n"])[:, 1])
        np.testing.assert_almost_equal(expected_count_wn, np.array(results["c_wn_n"])[:, 1].astype(float))
        np.testing.assert_equal(expected_count_ig, np.array(results["c_ig_t"])[:, 1])
        np.testing.assert_almost_equal(expected_count_wn, np.array(results["c_wn_t"])[:, 1].astype(float))

        # test base funcs - max/min
        expected_max_ig = [3, 3, 4, 4, 4, 5, 6, 7]
        expected_max_ig = [(datetime(2020, 1, 1) + timedelta(seconds=i + 3), expected_max_ig[i]) for i in range(8)]

        expected_max_wn = [3] + [float("nan")] * 6 + [7]
        self.assertEqual(expected_max_ig, results["mx_ig_n"])
        self.assertEqual(expected_max_ig, results["mx_ig_t"])
        self.assertEqual(expected_max_wn[0], results["mx_wn_n"][0][1])
        self.assertEqual(expected_max_wn[-1], results["mx_wn_n"][-1][1])
        self.assertEqual(expected_max_wn[0], results["mx_wn_t"][0][1])
        self.assertEqual(expected_max_wn[-1], results["mx_wn_t"][-1][1])
        for i in range(1, 6):
            self.assertTrue(math.isnan(np.array(results["mx_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["mx_wn_t"])[i, 1]))

        # test base funcs - sum
        expected_sum_ig = np.array([6, 5, 7, 4, 4, 5, 11, 18])
        np.testing.assert_equal(expected_sum_ig, np.array(results["s_ig_n"])[:, 1])
        np.testing.assert_equal(expected_sum_ig, np.array(results["s_ig_t"])[:, 1])
        for i in range(1, 7):
            self.assertTrue(math.isnan(np.array(results["s_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["s_wn_t"])[i, 1]))
        np.testing.assert_equal(18, np.array(results["s_wn_n"])[-1, 1])
        np.testing.assert_equal(18, np.array(results["s_wn_t"])[-1, 1])

        # test base funcs - mean
        expected_mean_ig = [2, 2.5, 3.5, 4, 4, 5, 5.5, 6]
        expected_mean_wn = [2] + [float("nan")] * 6 + [6]
        np.testing.assert_equal(expected_mean_ig, np.array(results["mu_ig_t"])[:, 1])
        np.testing.assert_equal(expected_mean_ig, np.array(results["mu_ig_n"])[:, 1])
        for i in range(1, 7):
            self.assertTrue(math.isnan(np.array(results["mu_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["mu_wn_t"])[i, 1]))
        np.testing.assert_equal(6, np.array(results["mu_wn_n"])[-1, 1])
        np.testing.assert_equal(6, np.array(results["mu_wn_t"])[-1, 1])

        # test base funcs - stddev
        expected_std_ig = [1, 0.5 ** (1 / 2), 0.5 ** (1 / 2), 0.5 ** (1 / 2), 1]  # excluding nans
        expected_std_wn = [1] + [float("nan")] * 3 + [1]
        for i in range(1, 7):
            self.assertTrue(math.isnan(np.array(results["std_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["std_wn_t"])[i, 1]))
        for i in range(3, 6):
            # only one data point => nan stddev
            self.assertTrue(math.isnan(np.array(results["std_ig_t"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["std_ig_n"])[i, 1]))

        np.testing.assert_almost_equal(1, np.array(results["std_wn_t"])[-1, 1], decimal=7)
        np.testing.assert_almost_equal(1, np.array(results["std_wn_n"])[-1, 1], decimal=7)
        # remove ddof nans from ignore
        ignore_t = np.array(results["std_ig_t"])[:, 1]
        ignore_t = [x for x in ignore_t if not math.isnan(x)]
        ignore_n = np.array(results["std_ig_n"])[:, 1]
        ignore_n = [x for x in ignore_n if not math.isnan(x)]
        np.testing.assert_almost_equal(expected_std_ig, ignore_t, decimal=7)
        np.testing.assert_almost_equal(expected_std_ig, ignore_n, decimal=7)

        # test base funcs - quantile
        series = pd.Series(my_curve)
        expected_qt_ig_n0 = [1.5, 2.5, 3.5, 4.0, 4.0, 5.0, 5.5, 5.5]
        expected_qt_ig_n1 = [2.0, 2.5, 3.5, 4.0, 4.0, 5.0, 5.5, 6.0]
        expected_qt_ig_n2 = [2.5, 2.5, 3.5, 4.0, 4.0, 5.0, 5.5, 6.5]
        np.testing.assert_equal(np.array(expected_qt_ig_n0), np.array(results["qt_ig_n0"])[:, 1])
        np.testing.assert_equal(np.array(expected_qt_ig_n1), np.array(results["qt_ig_n1"])[:, 1])
        np.testing.assert_equal(np.array(expected_qt_ig_n2), np.array(results["qt_ig_n2"])[:, 1])

        self.assertEqual(1.5, results["qt_wn_n0"][0][1])
        self.assertEqual(5.5, results["qt_wn_n0"][-1][1])
        self.assertEqual(2.0, results["qt_wn_n1"][0][1])
        self.assertEqual(6.0, results["qt_wn_n1"][-1][1])
        self.assertEqual(2.5, results["qt_wn_n2"][0][1])
        self.assertEqual(6.5, results["qt_wn_n2"][-1][1])
        for i in range(1, 6):
            self.assertTrue(math.isnan(results["qt_wn_n0"][i][1]))
            self.assertTrue(math.isnan(results["qt_wn_n1"][i][1]))
            self.assertTrue(math.isnan(results["qt_wn_n2"][i][1]))

        expected_med_ig = series.rolling(window=3, min_periods=3).median()
        np.testing.assert_equal(np.array(expected_qt_ig_n1), np.array(results["med_ig_n"])[:, 1])

    def test_nan_handling_ema(self):
        dvalues = [1, 2, 3, float("nan"), 5, 6, 7, float("nan"), float("nan"), 10, 11, 12]
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), dvalues[i]) for i in range(12)])

            # ema
            ema_ignore = csp.stats.ema(x, alpha=0.1, adjust=False, ignore_na=True)
            ema_withnan = csp.stats.ema(x, span=20, adjust=True, ignore_na=False)
            ema_var_ignore = csp.stats.ema_var(x, alpha=0.1, adjust=False, ignore_na=True)
            ema_std_withnan = csp.stats.ema_std(x, span=20, adjust=True, bias=True, ignore_na=False)

            # test with horizon
            ema_horizon_ignore = csp.stats.ema(x, alpha=0.1, adjust=True, ignore_na=True, horizon=3)
            ema_horizon_withnan = csp.stats.ema(x, alpha=0.1, adjust=True, ignore_na=False, horizon=3)

            csp.add_graph_output("ema_ign", ema_ignore)
            csp.add_graph_output("ema_wn", ema_withnan)
            csp.add_graph_output("ema_var_ign", ema_var_ignore)
            csp.add_graph_output("ema_std_wn", ema_std_withnan)

            csp.add_graph_output("ema_horizon_ignore", ema_horizon_ignore)
            csp.add_graph_output("ema_horizon_withnan", ema_horizon_withnan)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=1000))

        # ema and ema_var
        values = pd.Series(dvalues)
        pd_alpha = values.ewm(alpha=0.1, adjust=False, ignore_na=True)
        pd_alpha_mean = pd_alpha.mean()
        pd_alpha_var = pd_alpha.var(bias=False)
        pd_span = values.ewm(span=20, adjust=True, ignore_na=False)
        pd_span_mean = pd_span.mean()
        pd_span_var = pd_span.std(bias=True)

        np.testing.assert_almost_equal(np.array(pd_alpha_mean), np.array(results["ema_ign"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(pd_span_mean), np.array(results["ema_wn"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(
            np.array(pd_alpha_var)[1:], np.array(results["ema_var_ign"])[:, 1][1:], decimal=7
        )
        self.assertTrue(math.isnan(results["ema_var_ign"][0][1]))  # first value is nan
        np.testing.assert_almost_equal(np.array(pd_span_var)[:], np.array(results["ema_std_wn"])[:, 1], decimal=7)

        # test with horizon (window calc)
        def ema_adj_ignore_nan(data, alpha):
            # weight using relative positions
            ema, weights = 0, 0
            cur_weight = 1
            for obs in reversed(data):
                if math.isnan(obs):
                    continue
                ema += obs * cur_weight
                weights += cur_weight
                cur_weight *= 1 - alpha
            return ema / weights

        def ema_adj_with_nan(data, alpha):
            # weight using global positions
            ema, weights = 0, 0
            cur_weight = 1
            for obs in reversed(data):
                cur_weight *= 1 - alpha
                if not math.isnan(obs):
                    ema += obs * cur_weight
                    weights += cur_weight
            return ema / weights

        exp_ema_hz_ign = [1.0, 2.9 / 1.9]
        exp_ema_hz_wn = [1.0, 2.9 / 1.9]
        for i in range(3, len(dvalues) + 1):
            window = dvalues[i - 3 : i]
            exp_ema_hz_ign.append(ema_adj_ignore_nan(window, 0.1))
            exp_ema_hz_wn.append(ema_adj_with_nan(window, 0.1))

        np.testing.assert_almost_equal(
            np.array(exp_ema_hz_ign), np.array(results["ema_horizon_ignore"])[:, 1], decimal=7
        )
        np.testing.assert_almost_equal(
            np.array(exp_ema_hz_wn), np.array(results["ema_horizon_withnan"])[:, 1], decimal=7
        )

    def test_adv_ema(self):
        # Tests restricted lookback and time-based half lives
        my_data = [i for i in range(10)]
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=int, data=[(st + timedelta(seconds=i + 1), val) for i, val in enumerate(my_data)])

            ema = csp.stats.ema(x, 3, 0.1, adjust=True, horizon=3)
            ema_half = csp.stats.ema(x, 3, halflife=timedelta(seconds=1))
            ema_std_half_debias = csp.stats.ema_std(x, 3, halflife=timedelta(seconds=1))

            csp.add_graph_output("ema", ema)
            csp.add_graph_output("ema_half", ema_half)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

        # Test horizon
        exp_ema = [
            (0.081 * x[0] + 0.09 * x[1] + 0.1 * x[2]) / (0.081 + 0.09 + 0.1)
            for x in [[i, i + 1, i + 2] for i in range(8)]
        ]
        np.testing.assert_almost_equal(exp_ema, np.array(results["ema"])[:, 1], decimal=7)

        # Test half-life
        exp_emahalf = [
            sum(0.5 ** (i - k) * my_data[k] for k in range(i + 1)) / sum(0.5 ** (i - k) for k in range(i + 1))
            for i in range(2, 10)
        ]
        np.testing.assert_almost_equal(exp_emahalf, np.array(results["ema_half"])[:, 1], decimal=6)

        # Test for bug when halflife > interval
        @csp.graph
        def graph(hl_days: float):
            d = csp.curve(
                typ=np.ndarray,
                data=[
                    (datetime(2010, 1, 4, 7, 0), np.array([1.0, 2.0, 3.0])),
                    (datetime(2010, 1, 5, 7, 0), np.array([1.5, 2.5, 3.5])),
                    (datetime(2010, 1, 6, 7, 0), np.array([2.0, 3.0, 4.0])),
                ],
            )
            out = csp.stats.ema(d, halflife=timedelta(days=hl_days))
            csp.add_graph_output("out", out)

        r1 = csp.run(graph, 0.5, starttime=datetime(2010, 1, 4, 7, 0), endtime=datetime(2010, 1, 5, 16, 0))
        np.testing.assert_almost_equal(
            np.array(r1["out"], dtype=object)[0, 1], np.array([1.0, 2.0, 3.0], dtype=object), decimal=4
        )
        np.testing.assert_almost_equal(
            np.array(r1["out"], dtype=object)[1, 1], np.array([1.4, 2.4, 3.4], dtype=object), decimal=4
        )

        r2 = csp.run(graph, 1, starttime=datetime(2010, 1, 4, 7, 0), endtime=datetime(2010, 1, 5, 16, 0))
        np.testing.assert_almost_equal(
            np.array(r2["out"], dtype=object)[0, 1], np.array([1.0, 2.0, 3.0], dtype=object), decimal=4
        )
        np.testing.assert_almost_equal(
            np.array(r2["out"], dtype=object)[1, 1], np.array([1.3333, 2.3333, 3.3333], dtype=object), decimal=4
        )

        r3 = csp.run(graph, 1 + 1e-5, starttime=datetime(2010, 1, 4, 7, 0), endtime=datetime(2010, 1, 5, 16, 0))
        np.testing.assert_almost_equal(
            np.array(r3["out"], dtype=object)[0, 1], np.array([1.0, 2.0, 3.0], dtype=object), decimal=4
        )
        np.testing.assert_almost_equal(
            np.array(r3["out"], dtype=object)[1, 1], np.array([1.3333, 2.3333, 3.3333], dtype=object), decimal=4
        )

        r4 = csp.run(graph, 2, starttime=datetime(2010, 1, 4, 7, 0), endtime=datetime(2010, 1, 5, 16, 0))
        norm = 1 + 1 / math.sqrt(2)
        np.testing.assert_almost_equal(
            np.array(r4["out"], dtype=object)[0, 1], np.array([1.0, 2.0, 3.0], dtype=object), decimal=4
        )
        np.testing.assert_almost_equal(
            np.array(r4["out"], dtype=object)[1, 1],
            np.array(
                [
                    (1 / math.sqrt(2) * 1 + 1.5) / norm,
                    (1 / math.sqrt(2) * 2 + 2.5) / norm,
                    (1 / math.sqrt(2) * 3 + 3.5) / norm,
                ],
                dtype=object,
            ),
            decimal=4,
        )

        r5 = csp.run(graph, 5, starttime=datetime(2010, 1, 4, 7, 0), endtime=datetime(2010, 1, 6, 16, 0))
        k1 = 0.5 ** (0.2)
        k2 = 0.5 ** (0.4)
        norm1 = 1 + k1
        norm2 = norm1 + k2
        np.testing.assert_almost_equal(
            np.array(r5["out"], dtype=object)[0, 1], np.array([1.0, 2.0, 3.0], dtype=object), decimal=4
        )
        np.testing.assert_almost_equal(
            np.array(r5["out"], dtype=object)[1, 1],
            np.array([(k1 * 1 + 1.5) / norm1, (k1 * 2 + 2.5) / norm1, (k1 * 3 + 3.5) / norm1], dtype=object),
            decimal=4,
        )
        np.testing.assert_almost_equal(
            np.array(r5["out"], dtype=object)[2, 1],
            np.array(
                [(k2 * 1 + k1 * 1.5 + 2) / norm2, (k2 * 2 + k1 * 2.5 + 3) / norm2, (k2 * 3 + k1 * 3.5 + 4) / norm2],
                dtype=object,
            ),
            decimal=4,
        )

    def test_cov_corr(self):
        N = 300
        data1 = np.random.uniform(low=-100, high=100, size=(N,))
        data2 = np.random.uniform(low=-100, high=100, size=(N,))

        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), val) for i, val in enumerate(data1)])
            y = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), val) for i, val in enumerate(data2)])

            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=i + 1), np.array([val, val], dtype=float)) for i, val in enumerate(data1)
                ],
            )
            y_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=i + 1), np.array([val, val], dtype=float)) for i, val in enumerate(data2)
                ],
            )

            cov = csp.stats.cov(x, y, timedelta(seconds=100), min_window=timedelta(seconds=1))
            corr = csp.stats.corr(x, y, 100, min_window=1)
            np_cov = csp.stats.cov(x_np, y_np, timedelta(seconds=100), min_window=timedelta(seconds=1))
            np_corr = csp.stats.corr(x_np, y_np, 100, min_window=1)
            ema_cov = csp.stats.ema_cov(x, y, alpha=0.1)
            np_ema_cov = csp.stats.ema_cov(x_np, y_np, alpha=0.1)

            csp.add_graph_output("cov", cov)
            csp.add_graph_output("np_cov", np_cov)
            csp.add_graph_output("corr", corr)
            csp.add_graph_output("np_corr", np_corr)
            csp.add_graph_output("ema_cov", ema_cov)
            csp.add_graph_output("np_ema_cov", np_ema_cov)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=N))

        # Test cov
        s1 = pd.Series(data1)
        s2 = pd.Series(data2)

        exp_cov = s1.rolling(window=100, min_periods=2).cov(s2)
        cov_arr = np.stack(np.array(results["np_cov"], dtype=object)[:, 1][1:])
        corr_arr = np.stack(np.array(results["np_corr"], dtype=object)[:, 1][1:])
        ema_cov_arr = np.stack(np.array(results["np_ema_cov"], dtype=object)[:, 1][1:])

        np.testing.assert_almost_equal(np.array(exp_cov)[1:], np.array(results["cov"])[:, 1][1:], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_cov)[1:], cov_arr[:, 0], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_cov)[1:], cov_arr[:, 1], decimal=7)
        # Test corr
        exp_corr = s1.rolling(window=100, min_periods=2).corr(s2)
        np.testing.assert_almost_equal(np.array(exp_corr)[1:], np.array(results["corr"])[:, 1][1:], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_corr)[1:], corr_arr[:, 0], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_corr)[1:], corr_arr[:, 1], decimal=7)

        # Test EMA covariance
        exp_ema_cov = s1.ewm(alpha=0.1).cov(other=s2)
        np.testing.assert_almost_equal(np.array(exp_ema_cov)[1:], np.array(results["ema_cov"])[:, 1][1:], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_ema_cov)[1:], ema_cov_arr[:, 0], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_ema_cov)[1:], ema_cov_arr[:, 1], decimal=7)

        # make sure the first tick for all calcs is nan
        for calc in ["cov", "corr", "ema_cov"]:
            self.assertTrue(math.isnan(results[calc][0][1]))
        for calc in ["np_cov", "np_corr", "np_ema_cov"]:
            self.assertTrue(math.isnan(results[calc][0][1][0]))
            self.assertTrue(math.isnan(results[calc][0][1][1]))

    def test_irregularly_spaced_data(self):
        data1 = [i for i in range(10)]

        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(
                typ=float,
                data=[
                    (st + timedelta(seconds=1), data1[0]),
                    (st + timedelta(seconds=3), data1[1]),
                    (st + timedelta(seconds=4.6), data1[2]),
                    (st + timedelta(seconds=4.8), data1[3]),
                    (st + timedelta(seconds=5.1), data1[4]),
                    (st + timedelta(seconds=7.8), data1[5]),
                    (st + timedelta(seconds=7.82), data1[6]),
                    (st + timedelta(seconds=7.89), data1[7]),
                    (st + timedelta(seconds=9.1), data1[8]),
                    (st + timedelta(seconds=10.8), data1[9]),
                ],
            )
            y = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), i + 1) for i in range(100)])
            y_np = csp.curve(
                typ=np.ndarray,
                data=[(st + timedelta(seconds=i + 1), np.array([i + 1], dtype=float)) for i in range(100)],
            )
            trigger = csp.curve(typ=bool, data=[(st + timedelta(seconds=30), True), (st + timedelta(seconds=70), True)])
            trigger_partial = csp.curve(
                typ=bool, data=[(st + timedelta(seconds=2), True), (st + timedelta(seconds=70), True)]
            )

            mean = csp.stats.mean(x, interval=timedelta(seconds=3))
            stddev = csp.stats.stddev(x, interval=timedelta(seconds=3))
            ema_var = csp.stats.ema_var(x, alpha=0.1)

            # case: lots of useless values moving through the buffer
            sum_t = csp.stats.sum(y, interval=timedelta(seconds=3), trigger=trigger)
            sum_n = csp.stats.sum(y, 3, trigger=trigger)
            mean_t = csp.stats.mean(y, interval=timedelta(seconds=3), trigger=trigger)
            mean_n = csp.stats.mean(y, 3, trigger=trigger)
            mean_tp = csp.stats.mean(
                y, interval=timedelta(seconds=3), min_window=timedelta(seconds=1), trigger=trigger_partial
            )
            mean_np = csp.stats.mean(y, 3, 1, trigger=trigger_partial)

            numpy_mean = csp.stats.mean(y_np, 3, 1, trigger=trigger)
            numpy_mean_t = csp.stats.mean(y_np, 3, 1, trigger=trigger)
            numpy_mean_p = csp.stats.mean(y_np, 3, 1, trigger=trigger_partial)

            # case: irregular data with reset at partially full initial buffer
            reset_partial = csp.curve(
                typ=bool, data=[(st + timedelta(seconds=2), True), (st + timedelta(seconds=7), True)]
            )
            reset_mean_t = csp.stats.mean(
                x, interval=timedelta(seconds=3), min_window=timedelta(seconds=1), reset=reset_partial
            )
            reset_mean_n = csp.stats.mean(x, 3, 1, reset=reset_partial)

            csp.add_graph_output("mean", mean)
            csp.add_graph_output("std", stddev)
            csp.add_graph_output("ema_var", ema_var)
            csp.add_graph_output("sum_t", sum_t)
            csp.add_graph_output("sum_n", sum_n)
            csp.add_graph_output("mean_t", mean_t)
            csp.add_graph_output("mean_n", mean_n)
            csp.add_graph_output("mean_tp", mean_tp)
            csp.add_graph_output("mean_np", mean_np)
            csp.add_graph_output("numpy_mean", numpy_mean)
            csp.add_graph_output("numpy_mean_p", numpy_mean_p)
            csp.add_graph_output("numpy_mean_t", numpy_mean_t)
            csp.add_graph_output("reset_mean_t", reset_mean_t)
            csp.add_graph_output("reset_mean_n", reset_mean_n)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        s1 = pd.Series(data1)

        exp_mean = [0.5, 1.5, 2, 2.5, 4.5, 5, 5.5, 6.5, 7.5]
        np.testing.assert_almost_equal(np.array(exp_mean)[:], np.array(results["mean"])[:, 1], decimal=7)

        exp_var = [0.5, 0.5, 1.0, 5 / 3, 0.5, 1.0, 5 / 3, 5 / 3, 5 / 3]
        np.testing.assert_almost_equal(np.sqrt(np.array(exp_var)[:]), np.array(results["std"])[:, 1], decimal=7)

        # test unused data moving through the buffer - ensure "removals" is accurate
        exp_sum = [(st + timedelta(seconds=30), 29 * 3), (st + timedelta(seconds=70), 69 * 3)]
        exp_mean = [(st + timedelta(seconds=30), 29), (st + timedelta(seconds=70), 69)]
        self.assertEqual(exp_sum, results["sum_t"])
        self.assertEqual(exp_sum, results["sum_n"])
        self.assertEqual(exp_mean, results["mean_t"])
        self.assertEqual(exp_mean, results["mean_n"])

        # test partially full initial removals, and then long wait, and then another trigger (should not include 3rd data point in removal)
        exp_mean[0] = (st + timedelta(seconds=2), 1.5)
        self.assertEqual(exp_mean, results["mean_tp"])
        self.assertEqual(exp_mean, results["mean_np"])

        # test same functionality for Numpy
        exp_mean = [np.array([29]), np.array([69])]
        exp_mean_p = [np.array([1.5]), np.array([69])]
        for i in range(2):
            np.testing.assert_equal(exp_mean[i], np.array(results["numpy_mean"], dtype=object)[i, 1])
            np.testing.assert_equal(exp_mean[i], np.array(results["numpy_mean_t"], dtype=object)[i, 1])
            np.testing.assert_equal(exp_mean_p[i], np.array(results["numpy_mean_p"], dtype=object)[i, 1])

        # test resetting with irregularly spaced data and a partially full buffer at first reset
        exp_mean_reset_t = [0, 1, 1.5, 2, 2.5, 5, 5.5, 6, 6.5, 7.5]
        exp_mean_reset_n = [0, 1, 1.5, 2, 3, 5, 5.5, 6, 7, 8]
        np.testing.assert_almost_equal(
            np.array(exp_mean_reset_t)[:], np.array(results["reset_mean_t"])[:, 1], decimal=7
        )
        np.testing.assert_almost_equal(
            np.array(exp_mean_reset_n)[:], np.array(results["reset_mean_n"])[:, 1], decimal=7
        )

    def test_quantiles_median(self):
        data = np.random.randint(low=-100, high=100, size=(2000,)).astype(float)  # want duplicates
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), val) for i, val in enumerate(data)])

            low = csp.stats.quantile(
                x,
                interval=timedelta(seconds=120),
                quant=[0.25, 0.5, 0.75],
                min_window=timedelta(seconds=2),
                interpolate="lower",
            )
            lin = csp.stats.quantile(x, interval=120, quant=[0.1, 0.2, 0.6], min_window=2, interpolate="linear")
            mid = csp.stats.quantile(x, interval=120, quant=0.75, min_window=2, interpolate="midpoint")
            high = csp.stats.quantile(
                x, interval=timedelta(seconds=120), quant=0.25, min_window=timedelta(seconds=2), interpolate="higher"
            )
            near = csp.stats.quantile(
                x, interval=timedelta(seconds=120), quant=0.333, min_window=timedelta(seconds=2), interpolate="nearest"
            )
            median1 = csp.stats.median(x, interval=timedelta(seconds=120), min_window=timedelta(seconds=2))
            median2 = csp.stats.median(x, interval=173, min_window=3)

            csp.add_graph_output("q1_low", low[0])
            csp.add_graph_output("q2_low", low[1])
            csp.add_graph_output("q3_low", low[2])

            csp.add_graph_output("q1_lin", lin[0])
            csp.add_graph_output("q2_lin", lin[1])
            csp.add_graph_output("q3_lin", lin[2])

            csp.add_graph_output("q_high", high)
            csp.add_graph_output("q_mid", mid)
            csp.add_graph_output("q_near", near)
            csp.add_graph_output("median1", median1)
            csp.add_graph_output("median2", median2)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=2000))
        s = pd.Series(data)

        exp_q1_low = s.rolling(window=120, min_periods=2).quantile(0.25, "lower")
        exp_q2_low = s.rolling(window=120, min_periods=2).quantile(0.5, "lower")
        exp_q3_low = s.rolling(window=120, min_periods=2).quantile(0.75, "lower")

        exp_q1_lin = s.rolling(window=120, min_periods=2).quantile(0.1, "linear")
        exp_q2_lin = s.rolling(window=120, min_periods=2).quantile(0.2, "linear")
        exp_q3_lin = s.rolling(window=120, min_periods=2).quantile(0.6, "linear")

        exp_q_high = s.rolling(window=120, min_periods=2).quantile(0.25, "higher")
        exp_q_mid = s.rolling(window=120, min_periods=2).quantile(0.75, "midpoint")
        exp_q_near = s.rolling(window=120, min_periods=2).quantile(0.333, "nearest")
        exp_median1 = s.rolling(window=120, min_periods=2).median()  # even
        exp_median2 = s.rolling(window=173, min_periods=3).median()  # odd

        np.testing.assert_almost_equal(np.array(exp_q1_low)[1:], np.array(results["q1_low"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_q2_low)[1:], np.array(results["q2_low"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_q3_low)[1:], np.array(results["q3_low"])[:, 1], decimal=7)

        np.testing.assert_almost_equal(np.array(exp_q1_lin)[1:], np.array(results["q1_lin"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_q2_lin)[1:], np.array(results["q2_lin"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_q3_lin)[1:], np.array(results["q3_lin"])[:, 1], decimal=7)

        np.testing.assert_almost_equal(np.array(exp_q_high)[1:], np.array(results["q_high"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_q_mid)[1:], np.array(results["q_mid"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_q_near)[1:], np.array(results["q_near"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_median1)[1:], np.array(results["median1"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(np.array(exp_median2)[2:], np.array(results["median2"])[:, 1], decimal=7)

    def test_weighted_statistics(self):
        st = datetime(2020, 1, 1)
        y_data = [np.random.rand() for i in range(10)]
        x_data = [i + 1 for i in range(30)]
        w_data = np.random.uniform(low=1.1, high=10, size=(10,))

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), x_data[i]) for i in range(10)])
            x2 = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), x_data[i]) for i in range(30)])
            y = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), y_data[i]) for i in range(10)])
            x_int = csp.curve(typ=int, data=[(st + timedelta(seconds=i + 1), x_data[i]) for i in range(10)])

            weights1 = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), 2 * x_data[i]) for i in range(10)])
            weights2 = csp.curve(typ=float, data=[(st + timedelta(seconds=2 * i + 1), x_data[i]) for i in range(5)])
            weights3 = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), w_data[i]) for i in range(10)])

            # functions like rolling square here
            wsum = csp.stats.sum(x, interval=timedelta(seconds=3), min_window=timedelta(seconds=1), weights=weights1)
            wpsum = csp.stats.sum(
                x, interval=timedelta(seconds=3), min_window=timedelta(seconds=1), precise=True, weights=weights2
            )
            wmean = csp.stats.mean(x, interval=timedelta(seconds=3), min_window=timedelta(seconds=1), weights=weights1)

            # var, std, sem, cov, corr
            wvar = csp.stats.var(
                x, interval=timedelta(seconds=3), min_window=timedelta(seconds=3), ddof=0, weights=weights3
            )
            wstd = csp.stats.stddev(
                x, interval=timedelta(seconds=3), min_window=timedelta(seconds=3), ddof=0, weights=weights3
            )
            wsem = csp.stats.sem(
                x, interval=timedelta(seconds=3), min_window=timedelta(seconds=3), ddof=0, weights=weights3
            )
            wcov = csp.stats.cov(
                x, y, interval=timedelta(seconds=3), min_window=timedelta(seconds=3), ddof=0, weights=weights3
            )
            wcorr = csp.stats.corr(
                x, y, interval=timedelta(seconds=3), min_window=timedelta(seconds=3), weights=weights3
            )

            # kurtosis and skew - should converge to unweighted with linearly increasing weights
            wkurt = csp.stats.kurt(
                x2, interval=timedelta(seconds=10), min_window=timedelta(seconds=5), bias=False, weights=weights1
            )
            kurt = csp.stats.kurt(x2, interval=timedelta(seconds=10), min_window=timedelta(seconds=5), bias=False)
            wskew = csp.stats.skew(
                x2, interval=timedelta(seconds=10), min_window=timedelta(seconds=5), bias=True, weights=weights1
            )
            skew = csp.stats.skew(x2, interval=timedelta(seconds=10), min_window=timedelta(seconds=5), bias=True)

            # make sure ints work
            int_sum = csp.stats.sum(
                x_int, interval=timedelta(seconds=3), min_window=timedelta(seconds=1), weights=weights1
            )
            int_mean = csp.stats.mean(
                x_int, interval=timedelta(seconds=3), min_window=timedelta(seconds=1), weights=weights1
            )

            csp.add_graph_output("wsum", wsum)
            csp.add_graph_output("wpsum", wpsum)
            csp.add_graph_output("wmean", wmean)
            csp.add_graph_output("wsem", wsem)
            csp.add_graph_output("wvar", wvar)
            csp.add_graph_output("wcov", wcov)
            csp.add_graph_output("wcorr", wcorr)
            csp.add_graph_output("wstd", wstd)
            csp.add_graph_output("wkurt", wkurt)
            csp.add_graph_output("kurt", kurt)
            csp.add_graph_output("wskew", wskew)
            csp.add_graph_output("skew", skew)
            csp.add_graph_output("int_sum", int_sum)
            csp.add_graph_output("int_mean", int_mean)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=30))

        # test sum
        exp_sum1 = [2, 10] + [sum(2 * i**2 for i in range(k, k + 3)) for k in range(1, 9)]
        exp_sum1 = np.array(exp_sum1)
        weights2_exp = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        exp_sum2 = [1, 3] + [sum((j + 1) * weights2_exp[j] for j in range(i, i + 3)) for i in range(8)]
        exp_sum2 = np.array(exp_sum2)
        np.testing.assert_equal(exp_sum1, np.array(results["wsum"])[:, 1])
        np.testing.assert_equal(exp_sum1, np.array(results["int_sum"])[:, 1])
        np.testing.assert_equal(exp_sum2, np.array(results["wpsum"])[:, 1])

        # test mean
        exp_mean = [1, 5 / 3] + [
            sum(2 * i**2 for i in range(k, k + 3)) / sum(2 * i for i in range(k, k + 3)) for k in range(1, 9)
        ]
        np.testing.assert_almost_equal(exp_mean, np.array(results["wmean"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(exp_mean, np.array(results["int_mean"])[:, 1], decimal=7)

        # test moment-based statistics against np weighted cov
        def weighted_cov(x, y, weights, interval, ddof, total_span):
            return [
                np.cov(
                    [x[i] for i in range(j, j + interval)],
                    [y[i] for i in range(j, j + interval)],
                    aweights=[weights[i] for i in range(j, j + interval)],
                    ddof=ddof,
                )[0][1]
                for j in range(total_span)
            ]

        def weighted_var(x, weights, interval, ddof, total_span):
            return weighted_cov(x, x, weights, interval, ddof, total_span)

        def weighted_std(x, weights, interval, ddof, total_span):
            return np.sqrt(weighted_var(x, weights, interval, ddof, total_span))

        def weighted_corr(x, y, weights, interval, total_span):
            return weighted_cov(x, y, weights, interval, 0, total_span) / (
                weighted_std(x, weights, interval, 0, total_span) * weighted_std(y, weights, interval, 0, total_span)
            )  # only if all ticks are non-nan

        def weighted_sem(x, weights, interval, ddof, total_span):
            weight_sum = pd.Series(weights).rolling(window=3, min_periods=3).sum()[2:]
            return weighted_std(x, weights, interval, ddof, total_span) / np.sqrt(weight_sum - ddof)

        exp_wcov = weighted_cov(x_data, y_data, w_data, 3, 0, 8)
        exp_wvar = weighted_var(x_data, w_data, 3, 0, 8)
        exp_wstd = weighted_std(x_data, w_data, 3, 0, 8)
        exp_wsem = weighted_sem(x_data, w_data, 3, 0, 8)
        exp_wcorr = weighted_corr(x_data, y_data, w_data, 3, 8)

        np.testing.assert_almost_equal(exp_wcov, np.array(results["wcov"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(exp_wvar, np.array(results["wvar"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(exp_wstd, np.array(results["wstd"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(exp_wsem, np.array(results["wsem"])[:, 1], decimal=7)
        np.testing.assert_almost_equal(exp_wcorr, np.array(results["wcorr"])[:, 1], decimal=7)

        # ensure skew and kurt converge
        np.testing.assert_almost_equal(np.array(results["wkurt"])[-1, 1], np.array(results["kurt"])[-1, 1], decimal=5)
        np.testing.assert_almost_equal(np.array(results["wskew"])[-1, 1], np.array(results["skew"])[-1:, 1], decimal=5)

    def test_numerical_stability(self):
        st = datetime(2020, 1, 1)
        original_state = np.random.get_state()
        np.random.seed(20)  # case that will show small, positive floating-point accumulation errors
        n_samples = 100
        weights = np.random.uniform(low=0.1, high=100.0, size=(n_samples,))
        weight_func = lambda i: 0 if i // 10 % 2 else weights[i]

        @csp.graph
        def graph():
            weights = csp.curve(
                typ=float, data=[(st + timedelta(seconds=i + 1), weight_func(i)) for i in range(n_samples)]
            )

            weights_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=i + 1), np.array([weight_func(i), weight_func(i)], dtype=np.float64))
                    for i in range(n_samples)
                ],
            )

            # every 10 ticks, set the next 10 ticks to zero
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), 1) for i in range(n_samples)])
            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=i + 1), np.array([1.0, 1.0], dtype=np.float64)) for i in range(n_samples)
                ],
            )

            wmean = csp.stats.mean(x, weights=weights, min_window=1, interval=10)
            np_wmean = csp.stats.mean(x_np, weights=weights_np, min_window=1, interval=10)
            np_horizon_ema = csp.stats.ema(weights_np, horizon=10, adjust=True, alpha=0.1)
            np_horizon_ema_std = csp.stats.ema_std(weights_np, horizon=10, adjust=True, alpha=0.1)
            np_horizon_ema_std_bias = csp.stats.ema_std(weights_np, horizon=10, adjust=True, alpha=0.1, bias=True)
            csp.add_graph_output("wmean", wmean)
            csp.add_graph_output("np_wmean", np_wmean)
            csp.add_graph_output("np_horizon_ema", np_horizon_ema)
            csp.add_graph_output("np_horizon_ema_std", np_horizon_ema_std)
            csp.add_graph_output("np_horizon_ema_std_bias", np_horizon_ema_std_bias)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=n_samples))

        for i in range(1, 6):
            """
            if the floating-point accumulation is accounted for, every 20th value should be nan as the total weight is 0
            since the total weight, after enough error, does not go to zero but instead goes to some small positive value, you get a non-nan value
            """
            self.assertTrue(np.isnan(results["wmean"][i * 20 - 1][1]))
            self.assertTrue(np.isnan(results["np_wmean"][i * 20 - 1][1][0]))
            self.assertTrue(np.isnan(results["np_wmean"][i * 20 - 1][1][1]))

            self.assertEqual(results["np_horizon_ema"][i * 20 - 1][1][0], 0)
            self.assertEqual(results["np_horizon_ema_std"][i * 20 - 1][1][0], 0)
            self.assertEqual(results["np_horizon_ema_std_bias"][i * 20 - 1][1][0], 0)

        np.random.set_state(original_state)

    def test_numpy_statistics(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=i + 1), np.arange(i + 1, i + 11, dtype=np.float64).reshape(2, 5))
                    for i in range(10)
                ],
            )

            v_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (
                        st + timedelta(seconds=i + 1),
                        np.array([i + 1, -i, 2 * i], dtype=np.float64).reshape(
                            3,
                        ),
                    )
                    for i in range(10)
                ],
            )
            v2_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (
                        st + timedelta(seconds=i + 1),
                        np.array([-i, 2 * i, i + 1], dtype=np.float64).reshape(
                            3,
                        ),
                    )
                    for i in range(10)
                ],
            )
            weights = csp.curve(
                typ=np.ndarray,
                data=[(st + timedelta(seconds=i + 1), np.full((2, 5), i + 1, dtype=np.float64)) for i in range(10)],
            )
            weights_f = csp.curve(
                typ=int, data=[(st + timedelta(seconds=i + 1), i + 1) for i in range(10)]
            )  # int typed
            weights_np = csp.curve(
                typ=np.ndarray,
                data=[(st + timedelta(seconds=i + 1), np.array([i + 1, i + 1, i + 1], dtype=float)) for i in range(10)],
            )  # same weights but for the element-wise cases

            dq_np_time = _window_updates(x_np, timedelta(seconds=3), x_np, x_np, csp.null_ts(float), csp.null_ts(float))
            dq_np_tick = _window_updates(x_np, 3, x_np, x_np, csp.null_ts(float), csp.null_ts(float))
            additions_np_n = dq_np_tick.additions
            removals_np_n = dq_np_tick.removals
            additions_np_t = dq_np_time.additions
            removals_np_t = dq_np_time.removals

            # Without shape
            sum_np = csp.stats.sum(x_np, timedelta(seconds=3), precise=True)
            wsum_np = csp.stats.sum(x_np, timedelta(seconds=3), weights=weights)
            mean_np = csp.stats.mean(x_np, timedelta(seconds=3))
            wmean_np = csp.stats.mean(x_np, 3, weights=weights)

            # elementwise cov/corr - test node functionality, actual computation tested in test_weighted_statistics
            cov_np = csp.stats.cov(v_np, v2_np, 5)
            wcov_np = csp.stats.cov(v_np, v2_np, 5, ddof=0, weights=weights_np)
            corr_np = csp.stats.corr(v_np, v2_np, 5)
            wcorr_np = csp.stats.corr(v_np, v2_np, 5, weights=weights_np)

            # matrix statistics
            cov_matrix_np = csp.stats.cov_matrix(v_np, 5)
            wcov_matrix_np = csp.stats.cov_matrix(v_np, 5, ddof=0, weights=weights_f)
            corr_matrix_np = csp.stats.corr_matrix(v_np, 5)
            wcorr_matrix_np = csp.stats.corr_matrix(v_np, 5, weights=weights_f)

            # test window updates
            csp.add_graph_output("remove_np_n", removals_np_n)
            csp.add_graph_output("add_np_n", additions_np_n)
            csp.add_graph_output("remove_np_t", removals_np_t)
            csp.add_graph_output("add_np_t", additions_np_t)

            # test sum
            csp.add_graph_output("sum_np", sum_np)
            csp.add_graph_output("wsum_np", wsum_np)

            # test mean
            csp.add_graph_output("mean_np", mean_np)
            csp.add_graph_output("wmean_np", wmean_np)

            # test covariance/correlation pairwise
            csp.add_graph_output("cov_matrix_np", cov_matrix_np)
            csp.add_graph_output("wcov_matrix_np", wcov_matrix_np)
            csp.add_graph_output("corr_matrix_np", corr_matrix_np)
            csp.add_graph_output("wcorr_matrix_np", wcorr_matrix_np)

            # test covariance/correlation elementwise
            csp.add_graph_output("cov_np", cov_np)
            csp.add_graph_output("wcov_np", wcov_np)
            csp.add_graph_output("corr_np", corr_np)
            csp.add_graph_output("wcorr_np", wcorr_np)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=30))

        # test window updates
        expected_rem = [(st + timedelta(seconds=i), [np.arange(i - 3, i + 7).reshape(2, 5)]) for i in range(4, 11)]
        expected_add = [(st + timedelta(seconds=i + 1), [np.arange(i + 1, i + 11).reshape(2, 5)]) for i in range(10)]
        np.testing.assert_equal(expected_rem, results["remove_np_n"])
        np.testing.assert_equal(expected_add, results["add_np_n"])
        np.testing.assert_equal(expected_rem, results["remove_np_t"])
        np.testing.assert_equal(expected_add, results["add_np_t"])

        # test sum

        expected_sum = [
            (
                st + timedelta(seconds=i),
                np.sum([np.arange(k, k + 10).reshape(2, 5) for k in range(i - 2, i + 1)], axis=0),
            )
            for i in range(3, 11)
        ]
        np.testing.assert_equal(expected_sum, results["sum_np"])

        expected_wsum = [
            (
                st + timedelta(seconds=i),
                np.sum([k * np.arange(k, k + 10).reshape(2, 5) for k in range(i - 2, i + 1)], axis=0),
            )
            for i in range(3, 11)
        ]
        np.testing.assert_equal(expected_wsum, results["wsum_np"])

        # test mean
        expected_mean = [
            np.sum([1 / 3 * np.arange(k, k + 10).reshape(2, 5) for k in range(i - 2, i + 1)], axis=0)
            for i in range(3, 11)
        ]
        np.testing.assert_almost_equal(expected_mean, list_nparr_to_matrix(results["mean_np"]), decimal=7)
        expected_wmean = [
            np.sum([k / (3 * i - 3) * np.arange(k, k + 10).reshape(2, 5) for k in range(i - 2, i + 1)], axis=0)
            for i in range(3, 11)
        ]
        np.testing.assert_almost_equal(expected_wmean, list_nparr_to_matrix(results["wmean_np"]), decimal=7)

        # test covariance
        expected_cov = np.cov([[i + 1 for i in range(5)], [-i for i in range(5)], [2 * i for i in range(5)]])
        expected_wcov = [
            np.cov(
                [[i + 1 for i in range(j, j + 5)], [-i for i in range(j, j + 5)], [2 * i for i in range(j, j + 5)]],
                aweights=[k + 1 for k in range(j, j + 5)],
                ddof=0,
            )
            for j in range(6)
        ]
        expected_corr = np.corrcoef([[i + 1 for i in range(5)], [-i for i in range(5)], [2 * i for i in range(5)]])
        expected_wcorr = expected_cov / np.sqrt(np.outer(np.diag(expected_cov), np.diag(expected_cov)))
        for i in range(6):
            # pairwise cov/corr
            np.testing.assert_almost_equal(
                expected_cov, np.array(results["cov_matrix_np"], dtype=object)[i, 1], decimal=7
            )
            np.testing.assert_almost_equal(
                expected_wcov[i], np.array(results["wcov_matrix_np"], dtype=object)[i, 1], decimal=7
            )
            np.testing.assert_almost_equal(
                expected_corr, np.array(results["corr_matrix_np"], dtype=object)[i, 1], decimal=7
            )
            np.testing.assert_almost_equal(
                expected_wcorr, np.array(results["wcorr_matrix_np"], dtype=object)[i, 1], decimal=7
            )
            # elementwise cov/corr
            cov_val = np.array(results["cov_np"], dtype=object)[i, 1].astype(float)
            np.testing.assert_almost_equal(expected_cov[0, 1], cov_val[0], decimal=7)
            np.testing.assert_almost_equal(expected_cov[1, 2], cov_val[1], decimal=7)
            np.testing.assert_almost_equal(expected_cov[2, 0], cov_val[2], decimal=7)
            corr_val = np.array(results["corr_np"], dtype=object)[i, 1].astype(float)
            np.testing.assert_almost_equal(expected_corr[0, 1], corr_val[0], decimal=7)
            np.testing.assert_almost_equal(expected_corr[1, 2], corr_val[1], decimal=7)
            np.testing.assert_almost_equal(expected_corr[2, 0], corr_val[2], decimal=7)
            weighted_cov_val = np.array(results["wcov_np"], dtype=object)[i, 1].astype(float)
            np.testing.assert_almost_equal(expected_wcov[i][0, 1], weighted_cov_val[0], decimal=7)
            np.testing.assert_almost_equal(expected_wcov[i][1, 2], weighted_cov_val[1], decimal=7)
            np.testing.assert_almost_equal(expected_wcov[i][2, 0], weighted_cov_val[2], decimal=7)
            weighted_corr_val = np.array(results["wcorr_np"], dtype=object)[i, 1].astype(float)
            np.testing.assert_almost_equal(expected_wcorr[0, 1], weighted_corr_val[0], decimal=7)
            np.testing.assert_almost_equal(expected_wcorr[1, 2], weighted_corr_val[1], decimal=7)
            np.testing.assert_almost_equal(expected_wcorr[2, 0], weighted_corr_val[2], decimal=7)

    def test_numpy_shape_setting(self):
        # test case where the following occurs
        # let's say interval = 3 s. Then data and sampler tick at t = 1 s and then trigger ticks at t = 5 s
        # There are no additions to be pushed since the tick at t=1 is out of the window now. So the calc node tries to return a NaN array but it doesn't know what shape to make it.
        # hence, we need to tick out that data at t=1 regardless of whether or not trigger ticks so the comp knows the shape.

        @csp.graph
        def g():
            x = csp.curve(typ=np.ndarray, data=[(datetime(2020, 1, 2), np.array([1.0, 1.0]))])
            trigger = csp.curve(typ=bool, data=[(datetime(2020, 1, 6), True)])
            reset = csp.curve(typ=bool, data=[(datetime(2020, 1, 1), True)])

            sum_np_t = csp.stats.sum(x, timedelta(days=3), trigger=trigger, reset=reset)
            csp.add_graph_output("sum_np_t", sum_np_t)

        res = csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(days=7))
        np.testing.assert_equal(res["sum_np_t"][0][1], np.array([0.0, 0.0]))

    def test_numpy_nan_handling(self):
        # 3x1 vector
        my_curve = [
            np.array([i + 1, i + 1, i + 1], dtype=float).reshape(
                3,
            )
            for i in range(10)
        ]
        my_curve[4] = np.array([np.nan, np.nan, np.nan], dtype=float)
        my_curve[6] = np.array([np.nan, np.nan, np.nan], dtype=float)
        my_curve[6] = np.array([np.nan, np.nan, np.nan], dtype=float)
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(
                typ=np.ndarray, data=[(st + timedelta(seconds=i + 1), my_curve[i]) for i in range(len(my_curve))]
            )
            x_samp = csp.curve(
                typ=np.ndarray,
                data=[(st + timedelta(seconds=i + 1), my_curve[i]) for i in range(4)]
                + [(st + timedelta(seconds=6), my_curve[5])]
                + [(st + timedelta(seconds=i + 1), my_curve[i]) for i in range(7, 10)],
            )
            sampler = csp.curve(typ=bool, data=[(st + timedelta(seconds=i + 1), True) for i in range(10)])

            # test sum, mean, cov_matrix
            sum_t = csp.stats.sum(x, timedelta(seconds=3), ignore_na=True)
            sum_n = csp.stats.sum(x_samp, 3, ignore_na=False, sampler=sampler)
            mean_n = csp.stats.mean(x_samp, 3, ignore_na=True, sampler=sampler)
            mean_t = csp.stats.mean(x, timedelta(seconds=3), ignore_na=False)
            cov_t = csp.stats.cov_matrix(x_samp, timedelta(seconds=3), ddof=1, ignore_na=True, sampler=sampler)
            cov_n = csp.stats.cov_matrix(x, timedelta(seconds=3), ddof=1, ignore_na=False)

            csp.add_graph_output("sum_t", sum_t)
            csp.add_graph_output("sum_n", sum_n)
            csp.add_graph_output("mean_n", mean_n)
            csp.add_graph_output("mean_t", mean_t)
            csp.add_graph_output("cov_t", cov_t)
            csp.add_graph_output("cov_n", cov_n)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

        # Make sure ignore_na=True works
        expected_sum_ig = [6, 9, 7, 10, 6, 14, 17, 27]
        expected_sum_ig = [
            np.array([v, v, v]).reshape(
                3,
            )
            for v in expected_sum_ig
        ]
        np.testing.assert_equal(expected_sum_ig, list_nparr_to_matrix(np.array(results["sum_t"], dtype=object)))

        expected_mean_ig = [2, 3, 3.5, 5, 6, 7, 8.5, 9]
        expected_mean_ig = [
            np.array([v, v, v]).reshape(
                3,
            )
            for v in expected_mean_ig
        ]
        np.testing.assert_equal(expected_mean_ig, list_nparr_to_matrix(np.array(results["mean_n"], dtype=object)))

        # Make sure ignore_na=False works
        for i in range(2):
            np.testing.assert_equal(expected_sum_ig[i], np.array(results["sum_n"], dtype=object)[i, 1])
            np.testing.assert_equal(expected_mean_ig[i], np.array(results["mean_t"], dtype=object)[i, 1])

        for i in range(2, 7):
            self.assertTrue(np.isnan((np.array(results["sum_n"], dtype=object)[i, 1])).all())
            self.assertTrue(np.isnan((np.array(results["mean_t"], dtype=object)[i, 1])).all())

        np.testing.assert_equal(expected_sum_ig[7], np.array(results["sum_n"], dtype=object)[7, 1])
        np.testing.assert_equal(expected_mean_ig[7], np.array(results["mean_t"], dtype=object)[7, 1])

    def test_sampler_nan_handling(self):
        # can do the same nan test as before, only with sampled input
        my_curve = [1, 2, 3, float("nan"), 4, float("nan"), float("nan"), 5, 6, 7]
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            y_samp = csp.curve(
                typ=float,
                data=[
                    (st + timedelta(seconds=1), 1),
                    (st + timedelta(seconds=2), 2),
                    (st + timedelta(seconds=3), 3),
                    (st + timedelta(seconds=5), 4),
                    (st + timedelta(seconds=8), 5),
                    (st + timedelta(seconds=9), 6),
                    (st + timedelta(seconds=10), 7),
                ],
            )
            sampler = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), True) for i in range(10)])

            # Count
            c_ig_n = csp.stats.count(y_samp, 3, 1, True, sampler=sampler)
            c_wn_n = csp.stats.count(y_samp, 3, 1, False, sampler=sampler)
            c_ig_t = csp.stats.count(y_samp, timedelta(seconds=3), timedelta(seconds=1), True, sampler=sampler)
            c_wn_t = csp.stats.count(y_samp, timedelta(seconds=3), timedelta(seconds=1), False, sampler=sampler)
            csp.add_graph_output("c_ig_n", c_ig_n)
            csp.add_graph_output("c_wn_n", c_wn_n)
            csp.add_graph_output("c_ig_t", c_ig_t)
            csp.add_graph_output("c_wn_t", c_wn_t)

            # Sum
            s_ig_n = csp.stats.sum(y_samp, 3, 3, ignore_na=True, sampler=sampler)
            s_wn_n = csp.stats.sum(y_samp, 3, 3, precise=True, ignore_na=False, sampler=sampler)
            s_ig_t = csp.stats.sum(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), precise=True, ignore_na=True, sampler=sampler
            )
            s_wn_t = csp.stats.sum(y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False, sampler=sampler)
            csp.add_graph_output("s_ig_n", s_ig_n)
            csp.add_graph_output("s_wn_n", s_wn_n)
            csp.add_graph_output("s_ig_t", s_ig_t)
            csp.add_graph_output("s_wn_t", s_wn_t)

            # Max (no need to test min)
            mx_ig_n = csp.stats.max(y_samp, 3, 3, ignore_na=True, sampler=sampler)
            mx_wn_n = csp.stats.max(y_samp, 3, 3, ignore_na=False, sampler=sampler)
            mx_ig_t = csp.stats.max(y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True, sampler=sampler)
            mx_wn_t = csp.stats.max(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False, sampler=sampler
            )
            csp.add_graph_output("mx_ig_n", mx_ig_n)
            csp.add_graph_output("mx_wn_n", mx_wn_n)
            csp.add_graph_output("mx_ig_t", mx_ig_t)
            csp.add_graph_output("mx_wn_t", mx_wn_t)

            # mean
            mu_ig_n = csp.stats.mean(y_samp, 3, 3, ignore_na=True, sampler=sampler)
            mu_wn_n = csp.stats.mean(y_samp, 3, 3, ignore_na=False, sampler=sampler)
            mu_ig_t = csp.stats.mean(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True, sampler=sampler
            )
            mu_wn_t = csp.stats.mean(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False, sampler=sampler
            )
            csp.add_graph_output("mu_ig_n", mu_ig_n)
            csp.add_graph_output("mu_wn_n", mu_wn_n)
            csp.add_graph_output("mu_ig_t", mu_ig_t)
            csp.add_graph_output("mu_wn_t", mu_wn_t)

            # stddev
            std_ig_n = csp.stats.stddev(y_samp, 3, 3, ignore_na=True, sampler=sampler)
            std_wn_n = csp.stats.stddev(y_samp, 3, 3, ignore_na=False, sampler=sampler)
            std_ig_t = csp.stats.stddev(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True, sampler=sampler
            )
            std_wn_t = csp.stats.stddev(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False, sampler=sampler
            )
            csp.add_graph_output("std_ig_n", std_ig_n)
            csp.add_graph_output("std_wn_n", std_wn_n)
            csp.add_graph_output("std_ig_t", std_ig_t)
            csp.add_graph_output("std_wn_t", std_wn_t)

            # quantile
            qt_ig_n = csp.stats.quantile(y_samp, 3, [0.25, 0.5, 0.75], 3, "midpoint", sampler=sampler)
            qt_wn_n = csp.stats.quantile(y_samp, 3, [0.25, 0.5, 0.75], 3, "midpoint", ignore_na=False, sampler=sampler)
            med_ig_n = csp.stats.median(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=True, sampler=sampler
            )
            med_wn_n = csp.stats.median(
                y_samp, timedelta(seconds=3), timedelta(seconds=3), ignore_na=False, sampler=sampler
            )
            csp.add_graph_output("qt_ig_n0", qt_ig_n[0])
            csp.add_graph_output("qt_ig_n1", qt_ig_n[1])
            csp.add_graph_output("qt_ig_n2", qt_ig_n[2])
            csp.add_graph_output("qt_wn_n0", qt_wn_n[0])
            csp.add_graph_output("qt_wn_n1", qt_wn_n[1])
            csp.add_graph_output("qt_wn_n2", qt_wn_n[2])
            csp.add_graph_output("med_ig_n", med_ig_n)
            csp.add_graph_output("med_wn_n", med_wn_n)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        # test base funcs - count
        expected_count_ig = np.array([1, 2, 3, 2, 2, 1, 1, 1, 2, 3])
        expected_count_wn = np.array([1, 2, 3, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3])
        np.testing.assert_equal(expected_count_ig, np.array(results["c_ig_n"])[:, 1])
        np.testing.assert_equal(expected_count_wn, np.array(results["c_wn_n"])[:, 1].astype(float))
        np.testing.assert_equal(expected_count_ig, np.array(results["c_ig_t"])[:, 1])
        np.testing.assert_equal(expected_count_wn, np.array(results["c_wn_t"])[:, 1].astype(float))

        # test base funcs - max/min
        expected_max_ig = [3, 3, 4, 4, 4, 5, 6, 7]
        expected_max_ig = [(datetime(2020, 1, 1) + timedelta(seconds=i + 3), expected_max_ig[i]) for i in range(8)]

        expected_max_wn = [3] + [float("nan")] * 6 + [7]
        self.assertEqual(expected_max_ig, results["mx_ig_n"])
        self.assertEqual(expected_max_ig, results["mx_ig_t"])
        self.assertEqual(expected_max_wn[0], results["mx_wn_n"][0][1])
        self.assertEqual(expected_max_wn[-1], results["mx_wn_n"][-1][1])
        self.assertEqual(expected_max_wn[0], results["mx_wn_t"][0][1])
        self.assertEqual(expected_max_wn[-1], results["mx_wn_t"][-1][1])
        for i in range(1, 6):
            self.assertTrue(math.isnan(np.array(results["mx_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["mx_wn_t"])[i, 1]))

        # test base funcs - sum
        expected_sum_ig = np.array([6, 5, 7, 4, 4, 5, 11, 18])
        np.testing.assert_equal(expected_sum_ig, np.array(results["s_ig_n"])[:, 1])
        np.testing.assert_equal(expected_sum_ig, np.array(results["s_ig_t"])[:, 1])
        for i in range(1, 7):
            self.assertTrue(math.isnan(np.array(results["s_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["s_wn_t"])[i, 1]))
        np.testing.assert_equal(18, np.array(results["s_wn_n"])[-1, 1])
        np.testing.assert_equal(18, np.array(results["s_wn_t"])[-1, 1])

        # test base funcs - mean
        expected_mean_ig = [2, 2.5, 3.5, 4, 4, 5, 5.5, 6]
        expected_mean_wn = [2] + [float("nan")] * 6 + [6]
        np.testing.assert_equal(expected_mean_ig, np.array(results["mu_ig_t"])[:, 1])
        np.testing.assert_equal(expected_mean_ig, np.array(results["mu_ig_n"])[:, 1])
        for i in range(1, 7):
            self.assertTrue(math.isnan(np.array(results["mu_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["mu_wn_t"])[i, 1]))
        np.testing.assert_equal(6, np.array(results["mu_wn_n"])[-1, 1])
        np.testing.assert_equal(6, np.array(results["mu_wn_t"])[-1, 1])

        # test base funcs - stddev
        expected_std_ig = [1, 0.5 ** (1 / 2), 0.5 ** (1 / 2), 0.5 ** (1 / 2), 1]  # excluding nans
        expected_std_wn = [1] + [float("nan")] * 3 + [1]
        for i in range(1, 7):
            self.assertTrue(math.isnan(np.array(results["std_wn_n"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["std_wn_t"])[i, 1]))
        for i in range(3, 6):
            # only one data point => nan stddev
            self.assertTrue(math.isnan(np.array(results["std_ig_t"])[i, 1]))
            self.assertTrue(math.isnan(np.array(results["std_ig_n"])[i, 1]))

        np.testing.assert_almost_equal(1, np.array(results["std_wn_t"])[-1, 1], decimal=7)
        np.testing.assert_almost_equal(1, np.array(results["std_wn_n"])[-1, 1], decimal=7)
        # remove ddof nans from ignore
        ignore_t = np.array(results["std_ig_t"])[:, 1]
        ignore_t = [x for x in ignore_t if not math.isnan(x)]
        ignore_n = np.array(results["std_ig_n"])[:, 1]
        ignore_n = [x for x in ignore_n if not math.isnan(x)]
        np.testing.assert_almost_equal(expected_std_ig, ignore_t, decimal=7)
        np.testing.assert_almost_equal(expected_std_ig, ignore_n, decimal=7)

        # test base funcs - quantile
        series = pd.Series(my_curve)
        expected_qt_ig_n0 = [1.5, 2.5, 3.5, 4.0, 4.0, 5.0, 5.5, 5.5]
        expected_qt_ig_n1 = [2.0, 2.5, 3.5, 4.0, 4.0, 5.0, 5.5, 6.0]
        expected_qt_ig_n2 = [2.5, 2.5, 3.5, 4.0, 4.0, 5.0, 5.5, 6.5]
        np.testing.assert_equal(np.array(expected_qt_ig_n0), np.array(results["qt_ig_n0"])[:, 1])
        np.testing.assert_equal(np.array(expected_qt_ig_n1), np.array(results["qt_ig_n1"])[:, 1])
        np.testing.assert_equal(np.array(expected_qt_ig_n2), np.array(results["qt_ig_n2"])[:, 1])

        self.assertEqual(1.5, results["qt_wn_n0"][0][1])
        self.assertEqual(5.5, results["qt_wn_n0"][-1][1])
        self.assertEqual(2.0, results["qt_wn_n1"][0][1])
        self.assertEqual(6.0, results["qt_wn_n1"][-1][1])
        self.assertEqual(2.5, results["qt_wn_n2"][0][1])
        self.assertEqual(6.5, results["qt_wn_n2"][-1][1])
        for i in range(1, 6):
            self.assertTrue(math.isnan(results["qt_wn_n0"][i][1]))
            self.assertTrue(math.isnan(results["qt_wn_n1"][i][1]))
            self.assertTrue(math.isnan(results["qt_wn_n2"][i][1]))

        expected_med_ig = series.rolling(window=3, min_periods=3).median()
        np.testing.assert_equal(np.array(expected_qt_ig_n1), np.array(results["med_ig_n"])[:, 1])

    def test_bivariate_out_of_seq_nan_handling(self):
        # test when there are nan values at different ticks for x and y in a bivariate calc (should ignore the pair of values)

        x_values = [1, 2, float("nan"), 3, 4, 2, float("nan"), float("nan"), float("nan"), 1, 2, 4, 6, 3]
        y_values = [1, 2, 1, float("nan"), 5, 7, float("nan"), float("nan"), float("nan"), 1, 2, 1, 3, 7]
        weights = [1 for i in range(len(x_values))]  # equal weighting hits bug case and allows comparison to pandas

        def g():
            x = csp.curve(
                typ=float,
                data=[(datetime(2020, 1, 1) + timedelta(seconds=i), x_val) for i, x_val in enumerate(x_values)],
            )
            y = csp.curve(
                typ=float,
                data=[(datetime(2020, 1, 1) + timedelta(seconds=i), y_val) for i, y_val in enumerate(y_values)],
            )
            w = csp.curve(
                typ=float,
                data=[(datetime(2020, 1, 1) + timedelta(seconds=i), weight) for i, weight in enumerate(weights)],
            )
            w_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (datetime(2020, 1, 1) + timedelta(seconds=i), np.array([weight, weight], dtype=float))
                    for i, weight in enumerate(weights)
                ],
            )
            xy = csp.curve(
                typ=np.ndarray,
                data=[
                    (datetime(2020, 1, 1) + timedelta(seconds=i), np.array([x_val, y_values[i]], dtype=float))
                    for i, x_val in enumerate(x_values)
                ],
            )
            yx = csp.curve(
                typ=np.ndarray,
                data=[
                    (datetime(2020, 1, 1) + timedelta(seconds=i), np.array([y_val, x_values[i]], dtype=float))
                    for i, y_val in enumerate(y_values)
                ],
            )

            cov = csp.stats.cov(x, y, timedelta(seconds=3), min_window=timedelta(seconds=0))
            ema_cov = csp.stats.ema_cov(x, y, 1, alpha=0.1)
            np_cov = csp.stats.cov_matrix(xy, timedelta(seconds=3), min_window=timedelta(seconds=0))
            corr = csp.stats.corr(x, y, 3, min_window=1)
            np_corr = csp.stats.corr_matrix(xy, 3, min_window=1)
            wsem = csp.stats.sem(x, interval=4, min_window=1, weights=w)
            np_wsem = csp.stats.sem(xy, interval=4, min_window=1, weights=w_np)
            np_ema_cov = csp.stats.ema_cov(xy, yx, alpha=0.1)  # hits _sync_nan_np

            csp.add_graph_output("cov", cov)
            csp.add_graph_output("np_cov", np_cov)
            csp.add_graph_output("corr", corr)
            csp.add_graph_output("np_corr", np_corr)
            csp.add_graph_output("ema_cov", ema_cov)
            csp.add_graph_output("wsem", wsem)
            csp.add_graph_output("np_wsem", np_wsem)
            csp.add_graph_output("np_ema_cov", np_ema_cov)

        # 1. cov and corr
        res = csp.run(
            g, starttime=datetime(2020, 1, 1), endtime=datetime(2020, 1, 1) + timedelta(seconds=len(x_values))
        )
        x_ser = pd.Series(x_values)
        y_ser = pd.Series(y_values)
        pd_cov = x_ser.rolling(window=3, min_periods=1).cov(y_ser)
        pd_corr = x_ser.rolling(window=3, min_periods=1).corr(y_ser)

        self.assertTrue(np.allclose(np.array(res["cov"])[:, 1].astype(float), np.array(pd_cov), equal_nan=True))
        self.assertTrue(np.allclose(np.array(res["corr"])[:, 1].astype(float), np.array(pd_corr), equal_nan=True))
        for i in range(len(pd_cov)):
            cov_val = res["np_cov"][i][1][0][1]
            corr_val = res["np_corr"][i][1][0][1]
            self.assertTrue(np.isclose(cov_val, np.array(pd_cov)[i], equal_nan=True))
            self.assertTrue(np.isclose(corr_val, np.array(pd_corr)[i], equal_nan=True))

        # 2. ema_cov
        pd_ema_cov = x_ser.ewm(alpha=0.1).cov(other=y_ser)
        ema_cov_arr = np.stack(np.array(res["np_ema_cov"], dtype=object)[:, 1])

        self.assertTrue(np.allclose(np.array(res["ema_cov"])[:, 1].astype(float), np.array(pd_ema_cov), equal_nan=True))
        self.assertTrue(np.allclose(ema_cov_arr[:, 0], np.array(pd_ema_cov), equal_nan=True))
        self.assertTrue(np.allclose(ema_cov_arr[:, 1], np.array(pd_ema_cov), equal_nan=True))

        # 3. weighted sem (ensure proper weighting)
        pd_sem_x = x_ser.rolling(window=4, min_periods=1).sem(ddof=1)
        pd_sem_y = y_ser.rolling(window=4, min_periods=1).sem(ddof=1)

        self.assertTrue(np.allclose(np.array(res["wsem"])[:, 1].astype(float), np.array(pd_sem_x), equal_nan=True))
        for i in range(len(pd_sem_x)):
            sem_x, sem_y = list(res["np_wsem"][i][1])
            self.assertTrue(np.isclose(sem_x, np.array(pd_sem_x)[i], equal_nan=True))
            self.assertTrue(np.isclose(sem_y, np.array(pd_sem_y)[i], equal_nan=True))

    def test_resetter(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(
                typ=float,
                data=[(st + timedelta(seconds=i + 1), i + 1) for i in range(8)]
                + [(st + timedelta(seconds=9), float("nan")), (st + timedelta(seconds=10), 10)],
            )

            reset = csp.curve(
                typ=float,
                data=[
                    (st + timedelta(seconds=4.5), True),
                    (st + timedelta(seconds=4.7), True),
                    (st + timedelta(seconds=7.5), True),
                    (st + timedelta(seconds=9.5), True),
                ],
            )

            # Count
            count = csp.stats.count(x, 3, 1, True, reset=reset)
            csp.add_graph_output("count", count)
            # Sum
            sum_t = csp.stats.sum(x, timedelta(seconds=3), timedelta(seconds=1), True, reset=reset)
            csp.add_graph_output("sum_t", sum_t)
            # Mean
            mean = csp.stats.mean(x, 3, 1, True, reset=reset)
            csp.add_graph_output("mean", mean)
            # Median
            med = csp.stats.median(x, timedelta(seconds=3), timedelta(seconds=1), True, reset=reset)
            csp.add_graph_output("med", med)

            # Reset and trigger at the same time - should tick out with default values
            trigger = csp.curve(
                typ=bool,
                data=[
                    (st + timedelta(seconds=4.5), True),
                    (st + timedelta(seconds=7.5), True),
                    (st + timedelta(seconds=9.5), True),
                ],
            )
            sum_trig_t = csp.stats.sum(
                x, timedelta(seconds=3), timedelta(seconds=1), True, trigger=trigger, reset=reset
            )
            sum_trig_n = csp.stats.sum(x, 3, 1, True, trigger=trigger, reset=reset)
            csp.add_graph_output("sum_trig_t", sum_trig_t)
            csp.add_graph_output("sum_trig_n", sum_trig_n)

            # Reset, data and trigger at the same time - should tick out the ticked value
            reset2 = csp.curve(
                typ=float,
                data=[
                    (st + timedelta(seconds=4), True),
                    (st + timedelta(seconds=7), True),
                    (st + timedelta(seconds=9), True),
                ],
            )
            sum_tr_t = csp.stats.sum(x, timedelta(seconds=3), timedelta(seconds=1), True, reset=reset2)
            sum_tr_n = csp.stats.sum(x, 3, 1, True, reset=reset2)
            csp.add_graph_output("sum_tr_t", sum_tr_t)
            csp.add_graph_output("sum_tr_n", sum_tr_n)

            # Reset, data, trigger at same time in NumPy case
            x_np = csp.curve(
                typ=np.ndarray,
                data=[(st + timedelta(seconds=i + 1), np.array([i + 1, i + 1], dtype=float)) for i in range(8)]
                + [
                    (st + timedelta(seconds=9), np.array([float("nan"), float("nan")], dtype=float)),
                    (st + timedelta(seconds=10), np.array([10, 10], dtype=float)),
                ],
            )
            sum_tr_t_np = csp.stats.sum(x_np, timedelta(seconds=3), timedelta(seconds=1), reset=reset2)
            sum_tr_n_np = csp.stats.sum(x_np, 3, 1, reset=reset2)
            csp.add_graph_output("sum_tr_t_np", sum_tr_t_np)
            csp.add_graph_output("sum_tr_n_np", sum_tr_n_np)

            # Min - r,d,t at same time
            min_t = csp.stats.min(x, timedelta(seconds=3), timedelta(seconds=1), True, reset=reset2)
            min_n = csp.stats.min(x, timedelta(seconds=3), timedelta(seconds=1), True, reset=reset2)
            csp.add_graph_output("min_t", min_t)
            csp.add_graph_output("min_n", min_n)

            # case: reset occurs with partially full buffer, then the data loops around a bunch, then we trigger computation
            reset3 = csp.curve(typ=bool, data=[(st + timedelta(seconds=3), True)])
            trigger2 = csp.curve(typ=bool, data=[(st + timedelta(seconds=8), True)])
            sum_t2 = csp.stats.sum(x, timedelta(seconds=3), trigger=trigger2, reset=reset3)
            sum_n2 = csp.stats.sum(x, 3, trigger=trigger2, reset=reset3)
            csp.add_graph_output("sum_t2", sum_t2)
            csp.add_graph_output("sum_n2", sum_n2)

            # case: make sure it doesn't return reset values in additions
            reset4 = csp.curve(
                typ=bool,
                data=[
                    (st + timedelta(seconds=3), True),
                    (st + timedelta(seconds=3.5), True),
                    (st + timedelta(seconds=4), True),
                ],
            )
            trigger3 = csp.curve(typ=bool, data=[(st + timedelta(seconds=5), True)])
            sum_t3 = csp.stats.sum(x, timedelta(seconds=3), trigger=trigger3, reset=reset4)
            sum_n3 = csp.stats.sum(x, 3, trigger=trigger3, reset=reset4)
            csp.add_graph_output("sum_t3", sum_t3)
            csp.add_graph_output("sum_n3", sum_n3)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

        expected_count = [1, 2, 3, 3, 1, 2, 3, 1, 1, 1]
        expected_count = [(st + timedelta(seconds=i + 1), expected_count[i]) for i in range(10)]
        self.assertEqual(expected_count, results["count"])

        expected_sum = [1, 3, 6, 9, 5, 11, 18, 8, 8, 10]
        expected_sum = [(st + timedelta(seconds=i + 1), expected_sum[i]) for i in range(10)]
        self.assertEqual(expected_sum, results["sum_t"])

        expected_mean = [1, 1.5, 2, 3, 5, 5.5, 6, 8, 8, 10]
        expected_mean = [(st + timedelta(seconds=i + 1), expected_mean[i]) for i in range(10)]
        self.assertEqual(expected_mean, results["mean"])

        expected_med = expected_mean
        self.assertEqual(expected_med, results["med"])

        expected_sum_trig = [
            (st + timedelta(seconds=4.5), 0),
            (st + timedelta(seconds=7.5), 0),
            (st + timedelta(seconds=9.5), 0),
        ]
        self.assertEqual(expected_sum_trig, results["sum_trig_t"])
        self.assertEqual(expected_sum_trig, results["sum_trig_n"])

        expected_sum_tr = [1, 3, 6, 4, 9, 15, 7, 15, 0, 10]
        expected_sum_tr = [(st + timedelta(seconds=i + 1), expected_sum_tr[i]) for i in range(10)]
        self.assertEqual(expected_sum_tr, results["sum_tr_t"])
        self.assertEqual(expected_sum_tr, results["sum_tr_n"])

        # NumPy test
        expected_sum_tr_np = [1, 3, 6, 4, 9, 15, 7, 15, 0, 10]
        expected_sum_tr_np = [np.array([expected_sum_tr_np[i], expected_sum_tr_np[i]]) for i in range(10)]
        for i in range(10):
            np.testing.assert_equal(expected_sum_tr_np[i], results["sum_tr_t_np"][i][1])
            np.testing.assert_equal(expected_sum_tr_np[i], results["sum_tr_n_np"][i][1])

        # Min (max) test
        expected_min = [1, 1, 1, 4, 4, 4, 7, 7, float("nan"), 10]
        expected_min = [(st + timedelta(seconds=i + 1), expected_min[i]) for i in range(10)]
        self.assertEqual(expected_min[:-2], results["min_t"][:-2])
        self.assertEqual(expected_min[:-2], results["min_n"][:-2])
        self.assertTrue(math.isnan(results["min_t"][-2][1]))
        self.assertTrue(math.isnan(results["min_n"][-2][1]))
        self.assertEqual(expected_min[-1][1], results["min_t"][-1][1])
        self.assertEqual(expected_min[-1][1], results["min_n"][-1][1])

        # Reset before any triggering with long wait in between
        self.assertEqual(21, results["sum_t2"][0][1])
        self.assertEqual(21, results["sum_n2"][0][1])
        self.assertEqual(9, results["sum_t3"][0][1])
        self.assertEqual(9, results["sum_n3"][0][1])

    def test_engine_cycle_case(self):
        # Testing triggers and ticks at the same time but different engine cycles

        @csp.node
        def source() -> csp.Outputs(value=csp.ts[float], np_val=csp.ts[np.ndarray], trigger=csp.ts[float]):
            with csp.alarms():
                alarm = csp.alarm(float)
                alarm2 = csp.alarm(float)
            with csp.state():
                s_nticks = 0

            with csp.start():
                csp.schedule_alarm(alarm, timedelta(seconds=1), 1.0)
                csp.schedule_alarm(alarm, timedelta(seconds=1), 1.0)
                csp.schedule_alarm(alarm2, timedelta(seconds=0.5), 1.0)

            if csp.ticked(alarm2):
                csp.output(np_val=np.array([0.0]))  # so shape is known

            if csp.ticked(alarm):
                if s_nticks % 2 == 0:  # When time runs out, "trigger" ticks first ...
                    csp.output(trigger=alarm)
                else:  # whereas "value" ticks at the same "time" but in the next engine cycle
                    csp.output(value=alarm)
                    csp.output(np_val=np.array([1.0]))
                csp.schedule_alarm(alarm, timedelta(seconds=1), 1.0)
                s_nticks += 1

        @csp.graph
        def graph() -> csp.ts[float]:
            src = source()
            sum_t = csp.stats.sum(
                src.value, timedelta(seconds=3), min_window=timedelta(seconds=0.5), trigger=src.trigger
            )
            np_sum_t = csp.stats.sum(
                src.np_val, timedelta(seconds=3), min_window=timedelta(seconds=0.5), trigger=src.trigger
            )
            sum_n = csp.stats.sum(src.value, 3, 1, trigger=src.trigger)
            csp.add_graph_output("sum_t", sum_t)
            csp.add_graph_output("sum_n", sum_n)
            csp.add_graph_output("np_sum_t", np_sum_t)

        st = datetime(2020, 1, 1)
        results = csp.run(graph, starttime=st, endtime=timedelta(seconds=8), output_numpy=True)
        np.testing.assert_equal(results["sum_n"][1], [0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0])
        np.testing.assert_equal(results["sum_t"][1], [0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        exp_np = [[0.0], [1.0], [2.0], [2.0], [2.0], [2.0], [2.0], [2.0]]
        for i in range(8):
            np.testing.assert_equal(results["np_sum_t"][1][i], exp_np[i])

    def test_error_messages(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            s = csp.curve(typ=str, data=[(st + timedelta(seconds=i + 1), "i") for i in range(8)])

            str_sum = csp.stats.sum(s, 3)

        with self.assertRaises(TypeError):
            results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

    def test_numpy_first_last_prod_unique_gmean(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x_np = csp.curve(
                typ=Numpy1DArray[float],
                data=[
                    (st + timedelta(seconds=1), np.array([[1, 1], [1, 1]], dtype=float)),
                    (st + timedelta(seconds=2), np.array([[2, 2], [2, 2]], dtype=float)),
                    (st + timedelta(seconds=3), np.array([[float("nan"), 1], [float("nan"), 1]], dtype=float)),
                    (st + timedelta(seconds=4), np.array([[float("nan"), 1], [float("nan"), 1]], dtype=float)),
                    (st + timedelta(seconds=5), np.array([[3, 3], [3, 3]], dtype=float)),
                    (st + timedelta(seconds=6), np.array([[4, 4], [4, 4]], dtype=float)),
                ],
            )

            # first
            first_t = csp.stats.first(x_np, timedelta(seconds=2))
            first_n = csp.stats.first(x_np, 2)
            csp.add_graph_output("first_t", first_t)
            csp.add_graph_output("first_n", first_n)

            # last
            last_t = csp.stats.last(x_np, timedelta(seconds=2))
            last_n = csp.stats.last(x_np, 2)
            csp.add_graph_output("last_t", last_t)
            csp.add_graph_output("last_n", last_n)

            # prod
            prod_t = csp.stats.prod(x_np, timedelta(seconds=2))
            prod_n = csp.stats.prod(x_np, 2, ignore_na=False)
            csp.add_graph_output("prod_t", prod_t)
            csp.add_graph_output("prod_n", prod_n)

            # unique
            x_np2 = csp.curve(
                typ=Numpy1DArray[float],
                data=[
                    (st + timedelta(seconds=1), np.array([[1, 1], [1, 1]], dtype=np.float64)),
                    (st + timedelta(seconds=2), np.array([[2, 2], [2, 2]], dtype=np.float64)),
                    (
                        st + timedelta(seconds=3),
                        np.array([[float("nan"), float("nan")], [float("nan"), float("nan")]], dtype=np.float64),
                    ),
                    (st + timedelta(seconds=4), np.array([[2, 3], [2, 3]], dtype=np.float64)),
                    (st + timedelta(seconds=5), np.array([[4, 4], [4, 4]], dtype=np.float64)),
                ],
            )
            unq_t = csp.stats.unique(x_np2, timedelta(seconds=4), min_window=timedelta(seconds=1))
            csp.add_graph_output("unq_t", unq_t)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

        def test_2x2_stat(exp, act):
            v1 = exp[0]
            v2 = exp[1]
            exp = np.array([[v1, v2], [v1, v2]], dtype=float)
            np.testing.assert_almost_equal(exp, np.array(act, dtype=float), decimal=7)

        expected_first = [(1, 1), (2, 2), (float("nan"), 1), (3, 1), (3, 3)]
        expected_last_ig = [(2, 2), (2, 1), (float("nan"), 1), (3, 3), (4, 4)]
        expected_prod_ig = [(2, 2), (2, 2), (float("nan"), 1), (3, 3), (12, 12)]
        expected_prod_wn = [(2, 2), (float("nan"), 2), (float("nan"), 1), (float("nan"), 3), (12, 12)]
        expected_unq = [[1, 1, 1, 1], [2, 2, 2, 2], [2, 2, 2, 2], [2, 3, 2, 3], [2, 3, 2, 3]]
        expected_unq = [np.array([[v[0], v[1]], [v[2], v[3]]], dtype=float) for v in expected_unq]

        for i in range(5):
            test_2x2_stat(expected_first[i], results["first_t"][i][1])
            test_2x2_stat(expected_first[i], results["first_n"][i][1])
            test_2x2_stat(expected_last_ig[i], results["last_t"][i][1])
            test_2x2_stat(expected_last_ig[i], results["last_n"][i][1])
            test_2x2_stat(expected_prod_ig[i], results["prod_t"][i][1])
            test_2x2_stat(expected_prod_wn[i], results["prod_n"][i][1])

        np.testing.assert_almost_equal(
            expected_unq, list_nparr_to_matrix(np.array(results["unq_t"], dtype=object)), decimal=7
        )

    def test_numpy_min_max_qtl_median(self):
        st = datetime(2020, 1, 1)
        data = [np.random.uniform(low=-100, high=100, size=(2, 2)) for i in range(100)]

        @csp.graph
        def graph():
            x_np = csp.curve(typ=np.ndarray, data=[(st + timedelta(seconds=i + 1), data[i]) for i in range(100)])

            # min/max
            min_t = csp.stats.min(x_np, timedelta(seconds=8))
            max_t = csp.stats.max(x_np, timedelta(seconds=8))
            min_n = csp.stats.min(x_np, 8)
            max_n = csp.stats.max(x_np, 8)
            csp.add_graph_output("min_t", min_t)
            csp.add_graph_output("max_t", max_t)
            csp.add_graph_output("max_n", max_n)
            csp.add_graph_output("min_n", min_n)

            # quantiles and median
            med_t = csp.stats.median(x_np, timedelta(seconds=8))
            med_n = csp.stats.median(x_np, 8)
            qt = csp.stats.quantile(x_np, timedelta(seconds=8), quant=[0.25, 0.75, 0.9], interpolate="linear")
            csp.add_graph_output("med_t", med_t)
            csp.add_graph_output("med_n", med_n)
            csp.add_graph_output("q1", qt[0])
            csp.add_graph_output("q2", qt[1])
            csp.add_graph_output("q3", qt[2])

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        # min/max
        s, expected_min, expected_max = [], [], []
        for i in range(8, 100):
            window = [data[j] for j in range(i - 8, i)]
            expected_max = np.maximum.reduce([x for x in window])
            expected_min = np.minimum.reduce([x for x in window])
            expected_qt = np.quantile([x for x in window], [0.25, 0.75, 0.9], axis=0)
            np.testing.assert_almost_equal(expected_max, np.array(results["max_n"], dtype=object)[i - 8, 1], decimal=7)
            np.testing.assert_almost_equal(expected_min, np.array(results["min_t"], dtype=object)[i - 8, 1], decimal=7)
            np.testing.assert_almost_equal(expected_max, np.array(results["max_t"], dtype=object)[i - 8, 1], decimal=7)
            np.testing.assert_almost_equal(expected_min, np.array(results["min_n"], dtype=object)[i - 8, 1], decimal=7)
            # quantile
            np.testing.assert_almost_equal(expected_qt[0], np.array(results["q1"], dtype=object)[i - 8, 1], decimal=7)
            np.testing.assert_almost_equal(expected_qt[1], np.array(results["q2"], dtype=object)[i - 8, 1], decimal=7)
            np.testing.assert_almost_equal(expected_qt[2], np.array(results["q3"], dtype=object)[i - 8, 1], decimal=7)

    def test_numpy_var_std_sem_skew_kurt(self):
        st = datetime(2020, 1, 1)

        data = [np.random.uniform(low=-100, high=100, size=(2, 2)) for i in range(1000)]

        @csp.graph
        def graph():
            x_np = csp.curve(typ=np.ndarray, data=[(st + timedelta(seconds=i + 1), data[i]) for i in range(100)])

            weights = csp.curve(
                typ=np.ndarray,
                data=[(st + timedelta(seconds=i + 1), np.full((2, 2), i + 1, dtype=float)) for i in range(100)],
            )

            # var/std
            var = csp.stats.var(x_np, timedelta(seconds=8))
            wvar = csp.stats.var(x_np, timedelta(seconds=8), weights=weights)
            stddev = csp.stats.stddev(x_np, 8)
            csp.add_graph_output("var", var)
            csp.add_graph_output("wvar", wvar)
            csp.add_graph_output("stddev", stddev)

            # sem/skew/kurt
            sem = csp.stats.sem(x_np, timedelta(seconds=8))
            skew = csp.stats.skew(x_np, 80)
            kurt = csp.stats.kurt(x_np, timedelta(seconds=80))
            wskew = csp.stats.skew(x_np, 80, weights=weights)
            wkurt = csp.stats.kurt(x_np, timedelta(seconds=80), weights=weights)
            csp.add_graph_output("sem", sem)
            csp.add_graph_output("skew", skew)
            csp.add_graph_output("kurt", kurt)
            csp.add_graph_output("wskew", wskew)
            csp.add_graph_output("wkurt", wkurt)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        # Ensure convergence
        # print(results['wskew'][-1][1])
        # print(results['wkurt'][-1][1])
        # print(results['skew'][-1][1])
        # print(results['kurt'][-1][1])

        for i in range(8, 100):
            window = [data[j] for j in range(i - 8, i)]
            expected_var = np.var([x for x in window], axis=0, ddof=1)
            expected_std = np.std([x for x in window], axis=0, ddof=1)
            expected_sem = expected_std / math.sqrt(7)

            np.testing.assert_almost_equal(expected_var, np.array(results["var"], dtype=object)[i - 8, 1], decimal=7)
            np.testing.assert_almost_equal(expected_std, np.array(results["stddev"], dtype=object)[i - 8, 1], decimal=7)
            np.testing.assert_almost_equal(expected_sem, np.array(results["sem"], dtype=object)[i - 8, 1], decimal=7)

    def test_numpy_ewm(self):
        st = datetime(2020, 1, 1)

        elem11 = np.random.uniform(low=-100, high=100, size=(100))
        elem12 = np.random.uniform(low=-100, high=100, size=(100))
        elem21 = np.random.uniform(low=-100, high=100, size=(100))
        elem22 = np.random.uniform(low=-100, high=100, size=(100))

        elems = [elem11, elem12, elem21, elem22]

        @csp.graph
        def graph():
            x = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=j + 1), np.array([[elems[0][j], elems[1][j]], [elems[2][j], elems[3][j]]]))
                    for j in range(100)
                ],
            )

            ema = csp.stats.ema(x, 3, 0.1, adjust=False)
            ema_adj = csp.stats.ema(x, 3, 0.1, adjust=True)
            ema_var = csp.stats.ema_var(x, 3, 0.1, bias=True)
            ema_std = csp.stats.ema_std(x, 3, 0.1, bias=False)
            ema_time = csp.stats.ema(x, 3, halflife=timedelta(seconds=1))
            ema_var_time = csp.stats.ema_var(x, 3, halflife=timedelta(seconds=1))
            ema_horizoned = csp.stats.ema(x, 3, 0.1, horizon=10)

            csp.add_graph_output("ema", ema)
            csp.add_graph_output("ema_adj", ema_adj)
            csp.add_graph_output("ema_var", ema_var)
            csp.add_graph_output("ema_std", ema_std)
            csp.add_graph_output("ema_time", ema_time)
            csp.add_graph_output("ema_var_time", ema_var_time)
            csp.add_graph_output("ema_horizoned", ema_horizoned)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        # test ema
        expected_adjusted_ema = np.zeros(shape=(98, 2, 2))
        expected_std_ema = np.zeros(shape=(98, 2, 2))
        expected_stddev_ema = np.zeros(shape=(98, 2, 2))
        expected_time_ema = np.zeros(shape=(98, 2, 2))
        expected_time_ema_var = np.zeros(shape=(98, 2, 2))

        for i in range(4):
            values = pd.Series(elems[i])
            pd_std = values.ewm(alpha=0.1, adjust=False).mean()
            pd_adj = values.ewm(alpha=0.1, adjust=True).mean()
            pd_sd = values.ewm(alpha=0.1, adjust=True).std(bias=False)
            pd_half = values.ewm(alpha=0.5, adjust=True).mean()
            pd_half_var = values.ewm(alpha=0.5, adjust=True).var(bias=False)
            for j in range(2, 100):
                expected_adjusted_ema[j - 2, i // 2, i % 2] = pd_adj[j]
                expected_std_ema[j - 2, i // 2, i % 2] = pd_std[j]
                expected_stddev_ema[j - 2, i // 2, i % 2] = pd_sd[j]
                expected_time_ema[j - 2, i // 2, i % 2] = pd_half[j]
                expected_time_ema_var[j - 2, i // 2, i % 2] = pd_half_var[j]

        for i in range(98):
            np.testing.assert_almost_equal(
                expected_std_ema[i, :, :], np.array(results["ema"][i][1], dtype=object), decimal=7
            )
            np.testing.assert_almost_equal(
                expected_adjusted_ema[i, :, :], np.array(results["ema_adj"][i][1], dtype=object), decimal=7
            )
            np.testing.assert_almost_equal(
                expected_stddev_ema[i, :, :], np.array(results["ema_std"][i][1], dtype=object), decimal=7
            )
            np.testing.assert_almost_equal(
                expected_time_ema[i, :, :], np.array(results["ema_time"][i][1], dtype=object), decimal=7
            )
            np.testing.assert_almost_equal(
                expected_time_ema_var[i, :, :], np.array(results["ema_var_time"][i][1], dtype=object), decimal=7
            )

    def test_ragged_arrays_error(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([1, 1, 1, 1], dtype=float)),  # 4x1
                    (st + timedelta(seconds=2), np.array([[1, 1], [1, 1]], dtype=float)),  # 2x2
                ],
            )

            sum_t = csp.stats.sum(x_np, timedelta(seconds=3))
            csp.add_graph_output("sum_t", sum_t)

        with self.assertRaises(ValueError):
            results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))

    def test_unnatural_strides(self):
        st = datetime(2020, 1, 1)

        @csp.node
        def broadcast_to(arr: csp.ts[csp.typing.NumpyNDArray[float]], shape: tuple) -> csp.ts[np.ndarray]:
            if csp.ticked(arr):
                return np.broadcast_to(arr, shape)

        @csp.node
        def flip_both_axes(arr: csp.ts[csp.typing.NumpyNDArray[float]]) -> csp.ts[np.ndarray]:
            if csp.ticked(arr):
                return np.flip(np.flip(arr, axis=0), axis=1)

        @csp.graph
        def graph():
            x1 = np.array([[1, 0], [2, 1]], dtype=float)
            x2 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
            x_np = csp.curve(typ=np.ndarray, data=[(st + timedelta(seconds=1), x1), (st + timedelta(seconds=2), x1.T)])
            x_np2 = csp.curve(
                typ=np.ndarray, data=[(st + timedelta(seconds=1), x2[:, 0]), (st + timedelta(seconds=2), x2[:, 1])]
            )

            # add dimension of 0
            x_np3 = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.expand_dims(x2[:, 0], axis=0)),
                    (st + timedelta(seconds=2), np.expand_dims(x2[:, 1], axis=0)),
                ],
            )

            sum_t = csp.stats.sum(x_np, timedelta(seconds=3), min_window=timedelta(seconds=1))
            csp.add_graph_output("sum_t", sum_t)

            sum_n = csp.stats.sum(x_np2, 3, 1)
            csp.add_graph_output("sum_n", sum_n)

            sum_n2 = csp.stats.sum(x_np3, 3, 1)
            csp.add_graph_output("sum_n2", sum_n2)

            # add dimension of 0 that isn't the last one...
            bts = broadcast_to(
                csp.sample(csp.timer(timedelta(seconds=1)), csp.const(np.array([1.0, 2.0, 3.0]))), (3, 3)
            )
            bts2 = broadcast_to(
                csp.sample(csp.timer(timedelta(seconds=1)), csp.const(np.array([1.0, 2.0, 3.0]))), (2, 3, 3)
            )
            sum_bts = csp.stats.sum(bts, 3, min_window=1)
            sum_bts2 = csp.stats.sum(bts2, 3, min_window=1)
            csp.add_graph_output("sum_bts", sum_bts)
            csp.add_graph_output("sum_bts2", sum_bts2)

            # test negative strides
            inp_arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float).reshape((2, 3))
            flipped_ts = flip_both_axes(csp.sample(csp.timer(timedelta(seconds=1)), csp.const(inp_arr)))
            flipped_sum = csp.stats.sum(flipped_ts, 3, 1)
            csp.add_graph_output("flipped_sum", flipped_sum)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=10))
        exp_arr1 = [np.array([[1, 0], [2, 1]], dtype=float), np.array([[2, 2], [2, 2]], dtype=float)]
        exp_arr2 = [
            np.array([1, 0, 0], dtype=float).reshape(
                3,
            ),
            np.array([1, 1, 0], dtype=float).reshape(
                3,
            ),
        ]
        exp_arr_bts = np.broadcast_to(np.array([1.0, 2.0, 3.0]), (3, 3)) * 3
        exp_arr_bts2 = np.broadcast_to(exp_arr_bts, (2, 3, 3))
        exp_flipped = np.array([float(i) for i in range(6, 0, -1)]).reshape((2, 3)) * 3

        for i in range(2):
            np.testing.assert_equal(exp_arr1[i], results["sum_t"][i][1])
            np.testing.assert_equal(exp_arr2[i], results["sum_n"][i][1])
            np.testing.assert_equal(np.expand_dims(exp_arr2[i], axis=0), results["sum_n2"][i][1])

        for i in range(2, 10):
            np.testing.assert_equal(exp_arr_bts, results["sum_bts"][i][1])
            np.testing.assert_equal(exp_arr_bts2, results["sum_bts2"][i][1])
            np.testing.assert_equal(exp_flipped, results["flipped_sum"][i][1])

    def test_listbasket(self):
        st = datetime(2020, 1, 1)

        @csp.node
        def takes_1darray(x: csp.ts[csp.typing.Numpy1DArray[float]]) -> csp.ts[csp.typing.Numpy1DArray[float]]:
            return x

        @csp.graph
        def graph():
            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([1, 1], dtype=float)),
                    (st + timedelta(seconds=2), np.array([2, 2], dtype=float)),
                ],
            )
            x_list1 = csp.curve(typ=float, data=[(st + timedelta(seconds=1), 1), (st + timedelta(seconds=2), 2)])
            x_list2 = csp.curve(typ=float, data=[(st + timedelta(seconds=1), 3), (st + timedelta(seconds=2), 4)])

            as_np = csp.stats.list_to_numpy([x_list1, x_list2])
            as_list = csp.stats.numpy_to_list(x_np, 2)

            csp.add_graph_output("as_np", takes_1darray(as_np))
            csp.add_graph_output("as_list0", as_list[0])
            csp.add_graph_output("as_list1", as_list[1])

            # test where some arrays don't have ticks
            y1 = csp.curve(typ=float, data=[(st + timedelta(seconds=1), 1)])
            y2 = csp.curve(typ=float, data=[(st + timedelta(seconds=2), 1)])

            np_nan = csp.stats.list_to_numpy([y1, y2], fillna=False)
            np_hold = csp.stats.list_to_numpy([y1, y2], fillna=True)
            csp.add_graph_output("np_nan", np_nan)
            csp.add_graph_output("np_hold", np_hold)

            # test convert, then compute, then convert back
            sum_n = csp.stats.numpy_to_list(csp.stats.sum(as_np, 2, 1), 2)
            csp.add_graph_output("sum_n0", sum_n[0])
            csp.add_graph_output("sum_n1", sum_n[1])

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=3))

        exp_np_nan = [np.array([1, np.nan], dtype=float), np.array([np.nan, 1], dtype=float)]
        exp_np_hold = [np.array([1, np.nan], dtype=float), np.array([1, 1], dtype=float)]
        exp_sum0 = [1, 3]
        exp_sum1 = [3, 7]

        for i in range(2):
            np.testing.assert_almost_equal(exp_np_nan[i], results["np_nan"][i][1])
            np.testing.assert_almost_equal(exp_np_hold[i], results["np_hold"][i][1])
            self.assertEqual(i + 1, results["as_list0"][i][1])
            self.assertEqual(i + 1, results["as_list1"][i][1])
            self.assertEqual(exp_sum0[i], results["sum_n0"][i][1])
            self.assertEqual(exp_sum1[i], results["sum_n1"][i][1])
            np.testing.assert_equal(np.array([i + 1, i + 3], dtype=float), results["as_np"][i][1])

    def test_elementwise_nan_handling(self):
        st = datetime(2020, 1, 1)

        data = [np.random.uniform(low=-100, high=100, size=(2, 2)).astype(float) for i in range(100)]
        # randomly place nans elementwise
        for i in range(100):
            for j in range(2):
                for k in range(2):
                    if np.random.rand() < 0.2:
                        data[i][j][k] = np.nan

        @csp.graph
        def graph():
            x = csp.curve(typ=np.ndarray, data=[(st + timedelta(seconds=i + 1), data[i]) for i in range(100)])

            mean = csp.stats.mean(x, 10)
            median = csp.stats.median(x, 10)
            stddev = csp.stats.stddev(x, 10)
            max_t = csp.stats.max(x, timedelta(seconds=10))

            csp.add_graph_output("mean", mean)
            csp.add_graph_output("median", median)
            csp.add_graph_output("stddev", stddev)
            csp.add_graph_output("max_t", max_t)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        for i in range(2):
            for j in range(2):
                el_data = np.array(data, dtype=object)[:, i, j]
                for k in range(91):
                    window = np.array(el_data[k : k + 10], dtype=float)
                    exp_mean = np.nanmean(window, axis=0)
                    exp_med = np.nanmedian(window, axis=0)
                    exp_std = np.nanstd(window, ddof=1, axis=0)
                    exp_max = np.nanmax(window, axis=0)

                    np.testing.assert_almost_equal(exp_mean, results["mean"][k][1][i][j], decimal=7)
                    np.testing.assert_almost_equal(exp_med, results["median"][k][1][i][j], decimal=7)
                    np.testing.assert_almost_equal(exp_std, results["stddev"][k][1][i][j], decimal=7)
                    np.testing.assert_almost_equal(exp_max, results["max_t"][k][1][i][j], decimal=7)

    def test_min_data_points(self):
        st = datetime(2020, 1, 1)
        data = [1, 1, np.nan, 1, np.nan, 1, 1, 1, np.nan, np.nan]

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), data[i]) for i in range(10)])

            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([[1, 1], [1, 1]], dtype=float)),
                    (st + timedelta(seconds=2), np.array([[2, np.nan], [2, np.nan]], dtype=float)),
                ],
            )

            mean = csp.stats.mean(x, 4, min_data_points=3)
            median = csp.stats.median(x, 4, min_data_points=3)
            stddev = csp.stats.stddev(x, 4, min_data_points=3)
            max_t = csp.stats.max(x, timedelta(seconds=4), min_data_points=3)

            sum_np = csp.stats.sum(x_np, 2, min_data_points=2)
            min_np = csp.stats.min(x_np, timedelta(seconds=2), min_data_points=2)

            csp.add_graph_output("mean", mean)
            csp.add_graph_output("median", median)
            csp.add_graph_output("stddev", stddev)
            csp.add_graph_output("max_t", max_t)

            csp.add_graph_output("sum_np", sum_np)
            csp.add_graph_output("min_np", min_np)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        exp_result = np.array([1, np.nan, np.nan, 1, 1, 1, np.nan], dtype=float)
        exp_std = np.array([0, np.nan, np.nan, 0, 0, 0, np.nan], dtype=float)

        np.testing.assert_almost_equal(
            exp_result, np.array(results["mean"], dtype=object)[:, 1].astype(float), decimal=7
        )
        np.testing.assert_almost_equal(
            exp_result, np.array(results["median"], dtype=object)[:, 1].astype(float), decimal=7
        )
        np.testing.assert_almost_equal(
            exp_result, np.array(results["max_t"], dtype=object)[:, 1].astype(float), decimal=7
        )
        np.testing.assert_almost_equal(
            exp_std, np.array(results["stddev"], dtype=object)[:, 1].astype(float), decimal=7
        )

        # Test NumPy
        exp_np_sum = np.array([[3, np.nan], [3, np.nan]], dtype=float)
        exp_np_min = np.array([[1, np.nan], [1, np.nan]], dtype=float)
        np.testing.assert_almost_equal(exp_np_sum, np.array(results["sum_np"][0][1], dtype=float), decimal=7)
        np.testing.assert_almost_equal(exp_np_min, np.array(results["min_np"][0][1], dtype=float), decimal=7)

    def test_elementwise_weighting(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([1, 1], dtype=float)),
                    (st + timedelta(seconds=2), np.array([2, 2], dtype=float)),
                ],
            )

            w = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([2, 1], dtype=float)),
                    (st + timedelta(seconds=2), np.array([1, 2], dtype=float)),
                ],
            )

            x_long = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([2, 1], dtype=float)),
                    (st + timedelta(seconds=2), np.array([1, 2], dtype=float)),
                    (st + timedelta(seconds=3), np.array([1, 2], dtype=float)),
                ],
            )

            w_long = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([2, 1], dtype=float)),
                    (st + timedelta(seconds=2), np.array([1, 2], dtype=float)),
                    (st + timedelta(seconds=3), np.array([1, 2], dtype=float)),
                ],
            )

            wsum = csp.stats.sum(x, interval=2, weights=w)
            wsum_p = csp.stats.sum(x, interval=timedelta(seconds=2), precise=True, weights=w)
            wmean = csp.stats.mean(x, interval=2, weights=w)
            wvar = csp.stats.var(x, interval=2, weights=w)
            wskew = csp.stats.skew(x_long, interval=3, weights=w_long)
            wkurt = csp.stats.kurt(x_long, interval=3, weights=w_long)

            csp.add_graph_output("wsum", wsum)
            csp.add_graph_output("wsum_p", wsum_p)
            csp.add_graph_output("wmean", wmean)
            csp.add_graph_output("wvar", wvar)
            csp.add_graph_output("wskew", wskew)
            csp.add_graph_output("wkurt", wkurt)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        exp_sum = np.array([4, 5], dtype=float)
        exp_mean = np.array([4 / 3, 5 / 3], dtype=float)
        exp_var = np.array([1 / 3, 1 / 3], dtype=float)

        # print(results['wskew'])
        # print(results['wkurt'])

        np.testing.assert_almost_equal(exp_sum, np.array(results["wsum"][0][1], dtype=float), decimal=7)
        np.testing.assert_almost_equal(exp_sum, np.array(results["wsum_p"][0][1], dtype=float), decimal=7)
        np.testing.assert_almost_equal(exp_mean, np.array(results["wmean"][0][1], dtype=float), decimal=7)
        np.testing.assert_almost_equal(exp_var, np.array(results["wvar"][0][1], dtype=float), decimal=7)

    def test_rank_argmin_argmax(self):
        st = datetime(2020, 1, 1)
        dval = [5, 10, 5, 3, 8, 8, 1, 0, 4, 5.3] * 10
        dval_np = 100 * np.random.normal(
            size=(
                100,
                4,
            )
        )
        dval2 = [1, 1, 1, 1, 2, 1]

        # more data for testing rank
        dval3 = [10, 1, 5, 6, 5, 6, 5, 6, 1.2, 2.4] * 10
        dval_np3 = np.zeros(
            shape=(
                100,
                4,
            )
        )
        for i in range(100):
            for j in range(4):
                dval_np3[i][j] = dval3[i]

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), dval[i]) for i in range(100)])

            x_np = csp.curve(typ=np.ndarray, data=[(st + timedelta(seconds=i + 1), dval_np[i]) for i in range(100)])

            x2 = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), dval2[i]) for i in range(6)])

            x3 = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), dval3[i]) for i in range(100)])

            x3_np = csp.curve(typ=np.ndarray, data=[(st + timedelta(seconds=i + 1), dval_np3[i]) for i in range(100)])

            rank_low = csp.stats.rank(x, 5)
            rank_np = csp.stats.rank(x_np, timedelta(seconds=5))
            argmin = csp.stats.argmin(x, 5)
            argmax = csp.stats.argmax(x, 5)
            argmin_np = csp.stats.argmin(x_np, 5)
            argmax_np = csp.stats.argmax(x_np, 5)
            argmax_early = csp.stats.argmax(x, 5, return_most_recent=False)
            rank_dup = csp.stats.rank(x2, timedelta(seconds=4), min_window=timedelta(seconds=0.1))

            # test alternative rank methods: min, max, avg
            rank_min = csp.stats.rank(x3, 5, method="min")
            rank_avg = csp.stats.rank(x3, 5, method="avg")
            rank_max = csp.stats.rank(x3, 5, method="max")

            # numpy: test alternative rank methods: min, max, avg
            rank_min_np = csp.stats.rank(x3_np, 5, method="min")
            rank_avg_np = csp.stats.rank(x3_np, 5, method="avg")
            rank_max_np = csp.stats.rank(x3_np, 5, method="max")

            csp.add_graph_output("rank_low", rank_low)
            csp.add_graph_output("rank_np", rank_np)
            csp.add_graph_output("argmin", argmin)
            csp.add_graph_output("argmax", argmax)
            csp.add_graph_output("argmin_np", argmin_np)
            csp.add_graph_output("argmax_np", argmax_np)
            csp.add_graph_output("argmax_early", argmax_early)
            csp.add_graph_output("rank_dup", rank_dup)

            csp.add_graph_output("rank_min", rank_min)
            csp.add_graph_output("rank_avg", rank_avg)
            csp.add_graph_output("rank_max", rank_max)
            csp.add_graph_output("rank_min_np", rank_min_np)
            csp.add_graph_output("rank_avg_np", rank_avg_np)
            csp.add_graph_output("rank_max_np", rank_max_np)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        exp_rank_low = [3, 2, 0, 0, 2, 3] + [3, 4, 1, 0, 3, 2, 0, 0, 2, 3] * 9
        for i in range(96):
            window_np = dval_np[i : i + 5, :]
            window_x = dval[i : i + 5]
            exp_rank_np = np.argsort(np.argsort(window_np, axis=0), axis=0)[-1]  # this is how you take rank
            np.testing.assert_equal(exp_rank_low[i], results["rank_low"][i][1])
            np.testing.assert_equal(exp_rank_np, results["rank_np"][i][1])
            # test
            exp_min = np.argmin(window_x)
            exp_max = np.argmax(window_x)
            np.testing.assert_equal(st + timedelta(seconds=i + 1 + int(exp_max)), results["argmax_early"][i][1])
            # handle dups in recent case
            if 2 <= i % 10 <= 4:
                exp_max += 1

            np.testing.assert_equal(st + timedelta(seconds=i + 1 + int(exp_min)), results["argmin"][i][1])
            np.testing.assert_equal(st + timedelta(seconds=i + 1 + int(exp_max)), results["argmax"][i][1])
            # test NumPy argmin/max
            exp_max_np = np.array(
                [st + timedelta(seconds=i + 1 + int(np.argmax(window_np, axis=0)[j])) for j in range(4)],
                dtype=np.datetime64,
            )
            exp_min_np = np.array(
                [st + timedelta(seconds=i + 1 + int(np.argmin(window_np, axis=0)[j])) for j in range(4)],
                dtype=np.datetime64,
            )
            np.testing.assert_equal(exp_min_np, results["argmin_np"][i][1])
            np.testing.assert_equal(exp_max_np, results["argmax_np"][i][1])

        # test ranks with duplicates
        for i in range(6):
            if i == 4:
                np.testing.assert_equal(3, results["rank_dup"][i][1])
            else:
                np.testing.assert_equal(0, results["rank_dup"][i][1])

        # test ranks with the three grouping methods
        dval3 = [10, 1, 5, 6, 5, 6, 5, 6, 1.2, 2.4] * 10
        exp_rank_min = [1, 3, 0, 2, 0, 1, 4, 0, 3, 3] * 9 + [1, 3, 0, 2, 0, 1]
        exp_rank_avg = [1.5, 3.5, 1, 3, 0, 1, 4, 0, 3, 3] * 9 + [1.5, 3.5, 1, 3, 0, 1]
        exp_rank_max = [2, 4, 2, 4, 0, 1, 4, 0, 3, 3] * 9 + [2, 4, 2, 4, 0, 1]

        for i in range(96):
            # single-valued
            np.testing.assert_equal(exp_rank_min[i], results["rank_min"][i][1])
            np.testing.assert_equal(exp_rank_avg[i], results["rank_avg"][i][1])
            np.testing.assert_equal(exp_rank_max[i], results["rank_max"][i][1])
            # numpy
            np.testing.assert_equal(
                np.array([exp_rank_min[i] for j in range(4)], dtype=object), results["rank_min_np"][i][1]
            )
            np.testing.assert_equal(
                np.array([exp_rank_avg[i] for j in range(4)], dtype=object), results["rank_avg_np"][i][1]
            )
            np.testing.assert_equal(
                np.array([exp_rank_max[i] for j in range(4)], dtype=object), results["rank_max_np"][i][1]
            )

        # test nan handling with most/least recent ties
        st = datetime(2022, 1, 1)
        _x = np.array([74.34, np.nan, 74.36, 74.38, 74.38, 74.34, 74.33, np.nan, 74.34, 74.34, 74.34, 74.34, 74.36])

        @csp.graph
        def graph_argmin():
            x = csp.curve(typ=float, data=(np.array([datetime(2022, 1, 1) + timedelta(x) for x in range(len(_x))]), _x))
            argmin_leastrec = csp.stats.argmin(x, 5, min_window=1, return_most_recent=False)
            argmin_mostrec = csp.stats.argmin(x, 5, min_window=1, return_most_recent=True)
            argmax_leastrec = csp.stats.argmax(x, 5, min_window=1, return_most_recent=False)
            argmax_mostrec = csp.stats.argmax(x, 5, min_window=1, return_most_recent=True)
            csp.add_graph_output("argmin_lr", argmin_leastrec)
            csp.add_graph_output("argmin_mr", argmin_mostrec)
            csp.add_graph_output("argmax_lr", argmax_leastrec)
            csp.add_graph_output("argmax_mr", argmax_mostrec)

        results = csp.run(graph_argmin, starttime=st, endtime=st + timedelta(15))

        least_recent_min = [1, 1, 1, 1, 1, 6, 7, 7, 7, 7, 7, 9, 9]
        most_recent_min = [1, 1, 1, 1, 1, 6, 7, 7, 7, 7, 7, 12, 12]
        least_recent_max = [1, 1, 3, 4, 4, 4, 4, 4, 5, 6, 9, 9, 13]
        most_recent_max = [1, 1, 3, 4, 5, 5, 5, 5, 5, 10, 11, 12, 13]

        for i in range(len(_x)):
            np.testing.assert_equal(st + timedelta(least_recent_min[i] - 1), results["argmin_lr"][i][1])
            np.testing.assert_equal(st + timedelta(most_recent_min[i] - 1), results["argmin_mr"][i][1])
            np.testing.assert_equal(st + timedelta(least_recent_max[i] - 1), results["argmax_lr"][i][1])
            np.testing.assert_equal(st + timedelta(most_recent_max[i] - 1), results["argmax_mr"][i][1])

        # test rank-specific nan handling options
        _x = np.array([1, np.nan, 2, np.nan, 3, 2, 1, 1, np.nan, np.nan, np.nan])

        @csp.graph
        def graph_rank():
            x = csp.curve(typ=float, data=(np.array([st + timedelta(x) for x in range(len(_x))]), _x))
            x_np = csp.curve(
                typ=np.ndarray,
                data=(np.array([st + timedelta(x) for x in range(len(_x))]), np.expand_dims(_x, axis=0).T),
            )

            rank_min_keep = csp.stats.rank(x, interval=3, method="min", na_option="keep")
            rank_max_keep = csp.stats.rank(x_np, interval=3, method="max")  # default to keep
            rank_min_last = csp.stats.rank(x, interval=3, method="min", na_option="last")
            rank_max_last = csp.stats.rank(x_np, interval=3, method="max", na_option="last")
            csp.add_graph_output("rank_min_keep", rank_min_keep)
            csp.add_graph_output("rank_max_keep", rank_max_keep)
            csp.add_graph_output("rank_min_last", rank_min_last)
            csp.add_graph_output("rank_max_last", rank_max_last)

        results = csp.run(graph_rank, starttime=st, endtime=st + timedelta(len(_x)))
        exp_rank_min_keep = [1, np.nan, 1, 0, 0, 0, np.nan, np.nan, np.nan]
        exp_rank_max_keep = [1, np.nan, 1, 0, 0, 1, np.nan, np.nan, np.nan]
        exp_rank_min_last = [1, 0, 1, 0, 0, 0, 0, 0, np.nan]
        exp_rank_max_last = [1, 0, 1, 0, 0, 1, 1, 0, np.nan]

        for i in range(len(_x) - 2):
            np.testing.assert_equal(exp_rank_min_keep[i], results["rank_min_keep"][i][1])
            np.testing.assert_equal(exp_rank_max_keep[i], results["rank_max_keep"][i][1])
            np.testing.assert_equal(exp_rank_min_last[i], results["rank_min_last"][i][1])
            np.testing.assert_equal(exp_rank_max_last[i], results["rank_max_last"][i][1])

    def test_cross_sectional(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.count(csp.timer(timedelta(seconds=1), 1))
            x_np = csp.curve(
                typ=np.ndarray, data=[(st + timedelta(seconds=i), np.array([i, i], dtype=float)) for i in range(1, 101)]
            )
            cs = csp.stats.cross_sectional(x, 10)
            cs_t = csp.stats.cross_sectional(x, timedelta(seconds=10))
            cs_np = csp.stats.cross_sectional(x_np, timedelta(seconds=10))
            cs_as_np = csp.stats.cross_sectional(x, timedelta(seconds=10), as_numpy=True)
            cs_np_as_np = csp.stats.cross_sectional(x_np, timedelta(seconds=10), as_numpy=True)

            naive_mean = csp.apply(cs, lambda x: sum(x) / len(x), float)
            naive_np_mean = csp.apply(cs_np, lambda x: np.sum(x, axis=0) / len(x), np.ndarray)
            naive_mean_from_np = csp.apply(cs_as_np, lambda x: np.sum(x) / len(x), float)
            naive_np_mean_from_np = csp.apply(cs_np_as_np, lambda x: np.sum(x, axis=0) / len(x), np.ndarray)

            csp.add_graph_output("cs", cs)
            csp.add_graph_output("cs_t", cs_t)
            csp.add_graph_output("cs_np", cs_np, tick_count=50)
            csp.add_graph_output("cs_as_np", cs_as_np)
            csp.add_graph_output("cs_np_as_np", cs_np_as_np, tick_count=50)

            csp.add_graph_output("naive_mean", naive_mean)
            csp.add_graph_output("naive_np_mean", naive_np_mean)
            csp.add_graph_output("naive_mean_from_np", naive_mean_from_np)
            csp.add_graph_output("naive_np_mean_from_np", naive_np_mean_from_np)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        exp_cs = [(st + timedelta(seconds=i), [j for j in range(i + 1 - 10, i + 1)]) for i in range(10, 101)]
        exp_cs_np = [
            (st + timedelta(seconds=i), [np.array([j, j]) for j in range(i + 1 - 10, i + 1)]) for i in range(51, 101)
        ]
        exp_mean = [
            (st + timedelta(seconds=i), sum([j for j in range(i + 1 - 10, i + 1)]) / 10) for i in range(10, 101)
        ]
        exp_np_mean = [
            (
                st + timedelta(seconds=i),
                np.array(
                    [sum([j for j in range(i + 1 - 10, i + 1)]) / 10, sum([j for j in range(i + 1 - 10, i + 1)]) / 10]
                ),
            )
            for i in range(10, 101)
        ]

        exp_cs_as_np = [
            (st + timedelta(seconds=i), np.array([j for j in range(i + 1 - 10, i + 1)], dtype=float))
            for i in range(10, 101)
        ]
        exp_cs_np_as_np = [
            (st + timedelta(seconds=i), np.array([[j, j] for j in range(i + 1 - 10, i + 1)], dtype=float))
            for i in range(51, 101)
        ]  # 10x2 array

        # test list return type
        np.testing.assert_equal(results["cs"], exp_cs)
        np.testing.assert_equal(results["cs_t"], exp_cs)
        np.testing.assert_equal(results["cs_np"], exp_cs_np)

        # test numpy return type
        np.testing.assert_equal(results["cs_as_np"], exp_cs_as_np)
        np.testing.assert_equal(results["cs_np_as_np"], exp_cs_np_as_np)

        # assert means are equal in both list and numpy cases
        np.testing.assert_equal(results["naive_mean"], exp_mean)
        np.testing.assert_equal(results["naive_np_mean"], exp_np_mean)
        np.testing.assert_equal(results["naive_mean_from_np"], exp_mean)
        np.testing.assert_equal(results["naive_np_mean_from_np"], exp_np_mean)

        # test empty return - list and NumPy
        @csp.graph
        def graph2():
            x = csp.null_ts(float)
            x_np = csp.null_ts(np.ndarray)
            trigger = csp.timer(timedelta(seconds=1))
            cs = csp.stats.cross_sectional(x, 10, 1, trigger=trigger)
            cs_np = csp.stats.cross_sectional(x_np, 10, 1, trigger=trigger)
            cs_as_np = csp.stats.cross_sectional(x, 10, 1, trigger=trigger, as_numpy=True)
            cs_np_as_np = csp.stats.cross_sectional(x_np, 10, 1, trigger=trigger, as_numpy=True)

            csp.add_graph_output("cs", cs)
            csp.add_graph_output("cs_np", cs_np)
            csp.add_graph_output("cs_as_np", cs_as_np)
            csp.add_graph_output("cs_np_as_np", cs_np_as_np)

        results = csp.run(graph2, starttime=st, endtime=st + timedelta(seconds=5))

        exp_list_out = [(st + timedelta(seconds=i + 1), []) for i in range(5)]
        exp_np_out = [(st + timedelta(seconds=i + 1), np.array([])) for i in range(5)]

        # test list return type
        np.testing.assert_equal(results["cs"], exp_list_out)
        np.testing.assert_equal(results["cs_np"], exp_list_out)

        # test numpy return type
        np.testing.assert_equal(results["cs_as_np"], exp_np_out)
        np.testing.assert_equal(results["cs_np_as_np"], exp_np_out)

        # Edge case that was broken where interval is exact size of buffer
        def graph3():
            td = timedelta()
            x_np = csp.curve(
                csp.typing.Numpy1DArray[float],
                [(td, np.array([1.0, 1.0, 1.0])), (td, np.array([2.0, 2.0, 2.0])), (td, np.array([3.0, 3.0, 3.0]))],
            )
            x_f = csp.curve(float, [(td, 1.0), (td, 2.0), (td, 3.0)])
            csp.add_graph_output("x_np", csp.stats.cross_sectional(x_np, interval=2))
            csp.add_graph_output("x_f", csp.stats.cross_sectional(x_f, interval=2))

        results = csp.run(graph3, starttime=st, endtime=st)
        numpy.testing.assert_array_equal(
            [v[1] for v in results["x_np"]],
            [
                [np.array([1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0])],
                [np.array([2.0, 2.0, 2.0]), np.array([3.0, 3.0, 3.0])],
            ],
        )

        self.assertEqual([v[1] for v in results["x_f"]], [[1.0, 2.0], [2.0, 3.0]])

        # Case where we're copying an empty window buffer
        data = csp.curve(
            typ=float, data=[(st, 1.0), (st + timedelta(seconds=4), 2.0), (st + timedelta(seconds=5), 3.0)]
        )
        data_np = csp.curve(
            typ=np.ndarray,
            data=[
                (st, np.array([1.0, 2.0])),
                (st + timedelta(seconds=4), np.array([2.0, 3.0])),
                (st + timedelta(seconds=5), np.array([3.0, 4.0])),
            ],
        )

        trigger = csp.curve(
            typ=bool,
            data=[
                (st + timedelta(seconds=3), True),
                (st + timedelta(seconds=4), True),
                (st + timedelta(seconds=5), True),
                (st + timedelta(seconds=8), True),
            ],
        )

        res_float_list = csp.run(
            csp.stats.cross_sectional(data, interval=timedelta(seconds=3), trigger=trigger),
            starttime=st,
            endtime=st + timedelta(seconds=8),
        )
        res_float_array = csp.run(
            csp.stats.cross_sectional(data, interval=timedelta(seconds=3), trigger=trigger, as_numpy=True),
            starttime=st,
            endtime=st + timedelta(seconds=8),
        )

        res_np_as_list = csp.run(
            csp.stats.cross_sectional(data_np, interval=timedelta(seconds=3), trigger=trigger),
            starttime=st,
            endtime=st + timedelta(seconds=8),
        )
        res_np_as_np = csp.run(
            csp.stats.cross_sectional(data_np, interval=timedelta(seconds=3), trigger=trigger, as_numpy=True),
            starttime=st,
            endtime=st + timedelta(seconds=8),
        )

        np.testing.assert_equal(
            res_float_list[0],
            [
                (st + timedelta(seconds=3), []),
                (st + timedelta(seconds=4), [2.0]),
                (st + timedelta(seconds=5), [2.0, 3.0]),
                (st + timedelta(seconds=8), []),
            ],
        )
        np.testing.assert_equal(
            res_float_array[0],
            [
                (st + timedelta(seconds=3), np.array([], dtype=np.float64)),
                (st + timedelta(seconds=4), np.array([2.0], dtype=np.float64)),
                (st + timedelta(seconds=5), np.array([2.0, 3.0], dtype=np.float64)),
                (st + timedelta(seconds=8), np.array([], dtype=np.float64)),
            ],
        )

        np.testing.assert_equal(res_np_as_list[0][0], (st + timedelta(seconds=3), []))
        np.testing.assert_equal(res_np_as_list[0][1], (st + timedelta(seconds=4), [np.array([2.0, 3.0])]))
        np.testing.assert_equal(
            res_np_as_list[0][2], (st + timedelta(seconds=5), [np.array([2.0, 3.0]), np.array([3.0, 4.0])])
        )
        np.testing.assert_equal(res_np_as_list[0][3], (st + timedelta(seconds=8), []))

        np.testing.assert_equal(res_np_as_np[0][0], (st + timedelta(seconds=3), np.array([], dtype=np.float64)))
        np.testing.assert_equal(res_np_as_np[0][1], (st + timedelta(seconds=4), np.array([[2.0, 3.0]])))
        np.testing.assert_equal(res_np_as_np[0][2], (st + timedelta(seconds=5), np.array([[2.0, 3.0], [3.0, 4.0]])))
        np.testing.assert_equal(res_np_as_np[0][3], (st + timedelta(seconds=8), np.array([], dtype=np.float64)))

    def test_expanding_window(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.count(csp.timer(timedelta(seconds=1), 1))
            x_np = csp.curve(
                typ=np.ndarray, data=[(st + timedelta(seconds=i), np.array([i, i], dtype=float)) for i in range(1, 101)]
            )
            reset = csp.curve(typ=bool, data=[(st + timedelta(seconds=50), True)])

            ct = csp.stats.count(x)
            sum = csp.stats.sum(x)
            mean_np = csp.stats.mean(x_np)
            std_np = csp.stats.stddev(x_np)
            median_np = csp.stats.median(x_np)
            reset_corr = csp.stats.corr(x, csp.cast_int_to_float(x), reset=reset, min_data_points=3)
            reset_np_sum = csp.stats.sum(x_np, reset=reset)

            csp.add_graph_output("ct", ct)
            csp.add_graph_output("sum", sum)
            csp.add_graph_output("mean_np", mean_np)
            csp.add_graph_output("std_np", std_np)
            csp.add_graph_output("median_np", median_np)
            csp.add_graph_output("reset_corr", reset_corr)
            csp.add_graph_output("reset_np_sum", reset_np_sum)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=100))

        for i in range(1, 101):
            data = [j + 1 for j in range(i)]
            data_np = [np.array([data[j], data[j]], dtype=float) for j in range(i)]
            np.testing.assert_almost_equal(data[-1], results["ct"][i - 1][1])
            np.testing.assert_almost_equal(np.sum(data), results["sum"][i - 1][1])
            np.testing.assert_almost_equal(np.mean(data_np, axis=0), results["mean_np"][i - 1][1])
            np.testing.assert_almost_equal(np.std(data_np, axis=0, ddof=1), results["std_np"][i - 1][1])
            np.testing.assert_almost_equal(np.median(data_np, axis=0), results["median_np"][i - 1][1])

            # test reset within expanding window
            if i in [1, 2, 50, 51]:
                np.testing.assert_almost_equal(np.nan, results["reset_corr"][i - 1][1])
            else:
                np.testing.assert_almost_equal(1, results["reset_corr"][i - 1][1])

            if i < 50:
                np.testing.assert_almost_equal(np.sum(data_np, axis=0), results["reset_np_sum"][i - 1][1])
            else:
                np.testing.assert_almost_equal(np.sum(data_np[49:], axis=0), results["reset_np_sum"][i - 1][1])

    def test_null_data(self):
        # Test what happens when there is no data in the interval

        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            # Test what happens at start
            x = csp.null_ts(float)
            x_np = csp.null_ts(np.ndarray)
            trigger = csp.timer(timedelta(seconds=1), 1.0)
            reset = csp.timer(timedelta(seconds=2))

            # mean, weighted mean, stddev (covers var/cov/sem), skew, kurt, median (covers quantile/min/max), ema, ema_std (covers ema_cov/var)
            count = csp.stats.count(x, 10, 1, trigger=trigger)
            sum_ = csp.stats.sum(x, 10, 1, trigger=trigger)
            unq = csp.stats.unique(x, 10, 1, trigger=trigger)
            mu = csp.stats.mean(x, 10, 1, trigger=trigger)
            wmu1 = csp.stats.mean(x, 10, 1, trigger=trigger, weights=trigger)  # x is null, weights are not
            wmu2 = csp.stats.mean(trigger, 10, 1, trigger=trigger, weights=x)  # weights are null, x is not
            std = csp.stats.stddev(x, 10, 1, trigger=trigger)
            skew = csp.stats.skew(x, 10, 1, trigger=trigger, reset=reset)  # reset too
            kurt = csp.stats.kurt(x, 10, 1, trigger=trigger, reset=reset)
            med = csp.stats.median(x, 10, 1, trigger=trigger, reset=reset)
            ema = csp.stats.ema(x, halflife=timedelta(seconds=1), trigger=trigger, reset=reset)
            ema_std = csp.stats.ema_std(x, alpha=0.1, trigger=trigger, reset=reset)

            calcs = [count, sum_, unq, mu, wmu1, wmu2, std, skew, kurt, med, ema, ema_std]
            names = ["count", "sum_", "unq", "mu", "wmu1", "wmu2", "std", "skew", "kurt", "med", "ema", "ema_std"]
            for i, c in enumerate(calcs):
                csp.add_graph_output(names[i], c)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=5))
        exp_out_nan = [(st + timedelta(seconds=i + 1), float("nan")) for i in range(5)]
        exp_out_zero = [(st + timedelta(seconds=i + 1), 0) for i in range(5)]

        # all should just be nan (or zero) when triggered
        for name, calc in results.items():
            if name in ["count", "sum_", "unq"]:
                np.testing.assert_equal(calc, exp_out_zero)
            else:
                np.testing.assert_equal(calc, exp_out_nan)

        @csp.graph
        def graph2():
            # Test what happens mid interval if no data points are left
            x = csp.curve(
                typ=float,
                data=[(st + timedelta(seconds=1), 1), (st + timedelta(seconds=2), 1), (st + timedelta(seconds=6), 1)],
            )

            x_np = csp.curve(
                typ=np.ndarray,
                data=[
                    (st + timedelta(seconds=1), np.array([1.0, 1.0])),
                    (st + timedelta(seconds=2), np.array([1.0, 1.0])),
                    (st + timedelta(seconds=6), np.array([1.0, 1.0])),
                ],
            )

            trigger = csp.curve(typ=bool, data=[(st + timedelta(seconds=5), True)])

            # mean, stddev (covers var/cov/sem), skew, kurt, median (covers quantile/min/max)
            count = lambda u: csp.stats.count(u, timedelta(seconds=2), trigger=trigger)
            sum_ = lambda u: csp.stats.sum(u, timedelta(seconds=2), trigger=trigger)
            unq = lambda u: csp.stats.unique(u, timedelta(seconds=2), trigger=trigger)
            mu = lambda u: csp.stats.mean(u, timedelta(seconds=2), trigger=trigger)
            std = lambda u: csp.stats.stddev(u, timedelta(seconds=2), trigger=trigger)
            skew = lambda u: csp.stats.skew(u, timedelta(seconds=2), trigger=trigger)
            kurt = lambda u: csp.stats.kurt(u, timedelta(seconds=2), trigger=trigger)
            med = lambda u: csp.stats.median(u, timedelta(seconds=2), trigger=trigger)

            calcs = [count, sum_, unq, mu, std, skew, kurt, med]
            names = ["count", "sum_", "unq", "mu", "std", "skew", "kurt", "med"]
            for i, c in enumerate(calcs):
                csp.add_graph_output(names[i], c(x))
                csp.add_graph_output(names[i] + "_numpy", c(x_np))

        results = csp.run(graph2, starttime=st, endtime=st + timedelta(seconds=5))

        exp_float_nan = [(st + timedelta(seconds=5), float("nan"))]
        exp_float_zero = [(st + timedelta(seconds=5), 0)]
        exp_numpy_nan = [(st + timedelta(seconds=5), np.array([np.nan, np.nan], dtype=float))]
        exp_numpy_zero = [(st + timedelta(seconds=5), np.array([0.0, 0.0]))]

        # all should just be nan (or zero) when triggered
        for name, calc in results.items():
            if "_numpy" in name:
                if name in ["count_numpy", "sum__numpy", "unq_numpy"]:
                    np.testing.assert_equal(calc, exp_numpy_zero)
                else:
                    np.testing.assert_equal(calc, exp_numpy_nan)
            else:
                if name in ["count", "sum_", "unq"]:
                    np.testing.assert_equal(calc, exp_float_zero)
                else:
                    np.testing.assert_equal(calc, exp_float_nan)

    def test_recalc(self):
        st = datetime(2020, 1, 1)
        rand_data = np.random.uniform(low=0.1, high=100.0, size=(100,))
        data_with_gaps = [rand_data[i] if (i // 10) % 2 == 0 else 0 for i in range(100)]

        @csp.graph
        def g():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), data_with_gaps[i]) for i in range(100)])
            x_np = csp.stats.list_to_numpy([x, x, x])  # 3x1 array
            recalc = csp.timer(timedelta(seconds=20), True)

            mean_nr = csp.stats.mean(x, 10, 1)
            sum_nr = csp.stats.sum(x, timedelta(seconds=10), timedelta(seconds=0.1))
            var_nr = csp.stats.var(x_np, 10, 1, ddof=0)
            ema_nr = csp.stats.ema(x_np, alpha=0.1, horizon=10)

            mean_recalc = csp.stats.mean(x, 10, 1, recalc=recalc)
            sum_recalc = csp.stats.sum(x, timedelta(seconds=10), timedelta(seconds=0.1), recalc=recalc)
            var_recalc = csp.stats.var(x_np, 10, 1, ddof=0, recalc=recalc)
            ema_recalc = csp.stats.ema(x_np, alpha=0.1, horizon=10, recalc=recalc)

            nodes = {
                "mean_nr": mean_nr,
                "sum_nr": sum_nr,
                "var_nr": var_nr,
                "ema_nr": ema_nr,
                "mean_recalc": mean_recalc,
                "sum_recalc": sum_recalc,
                "var_recalc": var_recalc,
                "ema_recalc": ema_recalc,
            }
            for name, node in nodes.items():
                csp.add_graph_output(name, node)

        results = csp.run(g, starttime=st, endtime=timedelta(seconds=100))
        recalc_nodes_f = ["mean_recalc", "sum_recalc"]
        no_recalc_nodes_f = ["mean_nr", "sum_nr"]

        # Recalcs are triggered exactly when the interval is all-zero, so no fp-error
        for j in range(5):
            idx = j * 20 - 1
            for i, node in enumerate(recalc_nodes_f):
                self.assertEqual(results[node][idx][1], 0)
                np.testing.assert_almost_equal(
                    results[node][idx][1], results[no_recalc_nodes_f[i]][idx][1], decimal=6
                )  # close to the un-recalculated value

            np.testing.assert_equal(results["var_recalc"][idx][1], np.array([0, 0, 0], dtype=float))
            np.testing.assert_almost_equal(
                results["var_recalc"][idx][1], results["var_nr"][idx][1], decimal=6
            )  # close to the un-recalculated value

            np.testing.assert_equal(results["ema_recalc"][idx][1], np.array([0, 0, 0], dtype=float))
            np.testing.assert_almost_equal(
                results["ema_recalc"][idx][1], results["ema_nr"][idx][1], decimal=6
            )  # happens to be the same due to defaults

        ## Test combinations with nans, reset and sample ##
        val = [i + 1 for i in range(20)]
        val[3] = float("nan")
        val[10] = float("nan")

        @csp.graph
        def g():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i + 1), val[i]) for i in range(20)])
            x_np = csp.curve(
                typ=np.ndarray,
                data=[(st + timedelta(seconds=i + 1), np.array([val[i], val[i]], dtype=float)) for i in range(20)],
            )  # 2x1

            # Reset and recalc at the same time should function the same as reset by itself
            reset = csp.curve(typ=bool, data=[(st + timedelta(seconds=5), True)])
            recalc = csp.curve(typ=int, data=[(st + timedelta(seconds=5), 1)])  # type doesn't matter
            sum_n = csp.stats.sum(x, 3, reset=reset, recalc=recalc)
            sum_t = csp.stats.sum(x, timedelta(seconds=3), reset=reset, recalc=recalc)
            csp.add_graph_output("sum_n", sum_n)
            csp.add_graph_output("sum_t", sum_t)

            # Test on corr/cov, corr_matrix (bivariate)
            corr_n = csp.stats.corr(x, x, 10, 1, recalc=recalc)
            corr_t = csp.stats.corr(x, x, timedelta(seconds=10), timedelta(seconds=1), recalc=recalc)
            corr_matrix_n = csp.stats.corr_matrix(x_np, 10, 1, recalc=recalc)
            corr_matrix_t = csp.stats.corr_matrix(x_np, timedelta(seconds=10), timedelta(seconds=1), recalc=recalc)
            csp.add_graph_output("corr_n", corr_n)
            csp.add_graph_output("corr_t", corr_t)
            csp.add_graph_output("corr_matrix_n", corr_matrix_n)
            csp.add_graph_output("corr_matrix_t", corr_matrix_t)

        results = csp.run(g, starttime=st, endtime=timedelta(seconds=20))

        exp_sum = [(i + 2) * 3 for i in range(18)]
        for i in range(1, 4):
            exp_sum[i] -= 4
        for i in range(8, 11):
            exp_sum[i] -= 11
        exp_sum[2] = 5  # reset and recalc

        for i in range(18):
            self.assertEqual(results["sum_n"][i][1], exp_sum[i])
            self.assertEqual(results["sum_t"][i][1], exp_sum[i])

        for i in range(2, 20):
            np.testing.assert_almost_equal(results["corr_n"][i][1], 1, decimal=7)
            np.testing.assert_almost_equal(results["corr_t"][i][1], 1, decimal=7)
            np.testing.assert_almost_equal(
                results["corr_matrix_n"][i][1], np.array([[1, 1], [1, 1]], dtype=float), decimal=7
            )
            np.testing.assert_almost_equal(
                results["corr_matrix_t"][i][1], np.array([[1, 1], [1, 1]], dtype=float), decimal=7
            )

    def test_negative_variance_instability(self):
        # case where, after a long stream of equal values, the variance (due to floating-point error) becomes erroneously negative
        # this then causes stddev to error, since its taking the sqrt of a negative value
        # the same applies for EMA variance as well - a slightly negative result causes ema_std to error out
        st = datetime(2020, 1, 1)
        original_state = np.random.get_state()
        np.random.seed(17)  # need to actually get negative FP accumulation

        dval = np.random.uniform(low=1.1, high=100, size=(100,))
        data_with_gaps = [dval[i] if (i // 10) % 2 == 0 else 10 for i in range(100)]

        @csp.graph
        def g():
            x = csp.curve(typ=float, data=[(st + timedelta(seconds=i), data_with_gaps[i]) for i in range(100)])
            x_np = csp.stats.list_to_numpy([x, x, x])
            var = csp.stats.var(x, 10, 1, ddof=0)
            wvar = csp.stats.var(x, 10, 1, weights=x, ddof=0)
            std = csp.stats.stddev(x, 10, 1, ddof=0)
            wstd = csp.stats.stddev(x, 10, 1, weights=x, ddof=0)
            ema_var_f = csp.stats.ema_var(x, alpha=0.1, horizon=10)
            ema_std_f = csp.stats.ema_std(x, alpha=0.1, horizon=10)
            ema_var_np = csp.stats.ema_var(x_np, alpha=0.1, horizon=10)
            ema_std_np = csp.stats.ema_std(x_np, alpha=0.1, horizon=10)

            csp.add_graph_output("var", var)
            csp.add_graph_output("std", std)
            csp.add_graph_output("wvar", wvar)
            csp.add_graph_output("wstd", wstd)
            csp.add_graph_output("ema_var_f", ema_var_f)
            csp.add_graph_output("ema_std_f", ema_std_f)
            csp.add_graph_output("ema_var_np", ema_var_np)
            csp.add_graph_output("ema_std_np", ema_std_np)

        results = csp.run(g, starttime=st, endtime=timedelta(seconds=100))
        for j in range(5):
            idx = j * 20 - 1
            for node in ["var", "std", "wvar", "wstd", "ema_var_f", "ema_std_f"]:
                self.assertGreaterEqual(results[node][idx][1], 0)
            for node in ["ema_var_np", "ema_std_np"]:
                for v in results[node][idx][1]:
                    self.assertGreaterEqual(v, 0)

        np.random.set_state(original_state)

    def test_skew_kurt_corr_instability_issue(self):
        # Since we divide by variance to a power in the online skew/kurtosis/correlation formulas
        # we need to ensure the variance is significant (i.e. not slightly > 0) to avoid nonsensical results

        st = datetime(2020, 1, 1)

        @csp.graph
        def g():
            x = csp.curve(
                typ=float,
                data=[
                    (datetime(2020, 1, 1), 1),
                    (datetime(2020, 1, 2), 1 - 1e-11),
                    (datetime(2020, 1, 3), 1),
                    (datetime(2020, 1, 4), 1 - 1e-12),
                    (datetime(2020, 1, 5), 1 - 1e-14),
                    (datetime(2020, 1, 6), 1 - 1e-10),
                    (datetime(2020, 1, 7), 1 - 1e-12),
                    (datetime(2020, 1, 8), 1),
                ],
            )
            skew = csp.stats.skew(x, 5, 1)
            kurt = csp.stats.kurt(x, 5, 1)
            corr = csp.stats.corr(x, x, 5, 1)
            csp.add_graph_output("skew", skew)
            csp.add_graph_output("kurt", kurt)
            csp.add_graph_output("corr", corr)

        res = csp.run(g, starttime=st, endtime=timedelta(days=8), output_numpy=True)

        # All should be NaN, as the small differences < 1e-9 should be resolved as floating point error
        self.assertTrue(pd.Series(res["skew"][1]).isna().all())
        self.assertTrue(pd.Series(res["kurt"][1]).isna().all())
        self.assertTrue(pd.Series(res["corr"][1]).isna().all())

    def test_allow_non_overlapping_bivariate(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def g(allow_non_overlapping: bool):
            x = csp.curve(
                typ=float,
                data=[
                    (datetime(2020, 1, 1), 1),
                    (datetime(2020, 1, 2), 2),
                    (datetime(2020, 1, 4), 4),  # discarded
                    (datetime(2020, 1, 6), 6),  # discarded
                    (datetime(2020, 1, 7), 7),
                ],
            )
            y = csp.curve(
                typ=float,
                data=[
                    (datetime(2020, 1, 1), 1),
                    (datetime(2020, 1, 2), 2),
                    (datetime(2020, 1, 3), -1),  # discarded
                    (datetime(2020, 1, 5), -2),  # discarded
                    (datetime(2020, 1, 7), 7),
                ],
            )
            cov = csp.stats.cov(x, y, 5, 1, allow_non_overlapping=allow_non_overlapping)
            corr = csp.stats.corr(x, y, 5, 1, allow_non_overlapping=allow_non_overlapping)
            ema_cov = csp.stats.ema_cov(x, y, 1, alpha=0.1, allow_non_overlapping=allow_non_overlapping)
            csp.add_graph_output("cov", cov)
            csp.add_graph_output("corr", corr)
            csp.add_graph_output("ema_cov", ema_cov)

        res = csp.run(g, True, starttime=st, endtime=timedelta(days=8), output_numpy=True)
        for output in res.values():
            # Convert expected datetimes to the same type as output
            expected_dates = pd.to_datetime([datetime(2020, 1, i) for i in (1, 2, 7)])
            np.testing.assert_array_equal(output[0], expected_dates.values)

        # Additional sanity check is that correlation should be 1 as middle ticks are ignored
        np.testing.assert_allclose(res["corr"][1], np.array([np.nan, 1.0, 1.0]), equal_nan=True)

    def test_ema_cov_horizon_bug(self):
        # Bug in finite horizon, adjusted, unbiased EMA covariance with ignore_na=True
        # Also applies to ema_var/ema_std as well, as they use cov
        # When the first data points are NaN, after the initial NaN is removed the next value that is removed has the wrong lookback weight applied to it

        st = datetime(2020, 1, 1)
        N = 15
        K = 3
        horizon = 10
        alpha = 0.1
        values = [np.nan if i < K else float(i) for i in range(N)]

        @csp.graph
        def g():
            x = csp.curve(
                typ=float, data=[(st + timedelta(seconds=i), values[i]) for i in range(N)]
            )  # start with some NaNs
            ema_std = csp.stats.ema_std(x, alpha=alpha, adjust=True, bias=False, ignore_na=True, horizon=horizon)
            csp.add_graph_output("ema_std", ema_std)

        res = csp.run(g, starttime=st, endtime=timedelta(seconds=N), output_numpy=True)

        golden_ema_std = np.array(
            [
                pd.Series(values[max(0, j - horizon + 1) : j + 1]).ewm(alpha=alpha, ignore_na=True).std().iloc[-1]
                for j in range(N)
            ]
        )
        np.testing.assert_allclose(res["ema_std"][1], golden_ema_std, atol=1e-10)

    def test_identical_values_variance(self):
        """Test that variance and weighted variance are exactly 0 when all values in window are identical"""
        st = datetime(2023, 1, 1, 9, 0, 0)

        K = 10
        N = 20

        def get_val(u):
            return float(int(u) // K)

        @csp.graph
        def graph():
            # Generate data with identical values in each 10 second window, 20 times to get some error accumulated
            data = csp.curve(float, [(st + timedelta(seconds=i + 1), get_val(i)) for i in range(K * N)])

            # Generate random weights
            w = np.random.uniform(low=0.1, high=1.0, size=K * N)
            weights = csp.curve(float, [(st + timedelta(seconds=i + 1), w[i]) for i in range(K * N)])

            # Test variance and weighted variance with 10-second resampling
            resample_interval = timedelta(seconds=K)
            timer = csp.timer(interval=resample_interval, value=True)

            # Regular variance
            var_result = csp.stats.var(data, interval=resample_interval, trigger=timer)

            # Weighted variance (using uniform weights of 1.0)
            wvar_result = csp.stats.var(data, interval=resample_interval, trigger=timer, weights=weights)

            csp.add_graph_output("variance", var_result)
            csp.add_graph_output("weighted_variance", wvar_result)

        results = csp.run(graph, starttime=st, endtime=timedelta(seconds=K * N), output_numpy=True)

        # Assert 1: all values in results['variance'] should be exactly 0, with no error
        np.testing.assert_equal(results["variance"][1], np.zeros(shape=(N,)))

        # Assert 2: weighted variance should equal unweighted variance
        np.testing.assert_equal(results["variance"], results["weighted_variance"])

    def test_unadjusted_ewm_halflife(self):
        import polars as pl

        N_DATA_POINTS = 100
        N_ELEM_NP = 3
        # polars does not ignore nan's in its ewm_mean_by (so need to generate data w/o nans)
        times, values = generate_random_data(N_DATA_POINTS, mu=0, sigma=1, pnan=0)

        @csp.graph
        def graph():
            test_data_float = csp.curve(typ=float, data=(times, values))
            test_data_np = csp.curve(
                typ=np.ndarray,
                data=(times, np.array([np.array([values[k] for j in range(N_ELEM_NP)]) for k in range(N_DATA_POINTS)])),
            )

            # EMA: float/np
            float_ema = csp.stats.ema(test_data_float, halflife=timedelta(seconds=5), adjust=False)
            np_ema = csp.stats.ema(test_data_np, halflife=timedelta(seconds=5), adjust=False)
            csp.add_graph_output("float_ema", float_ema)
            csp.add_graph_output("np_ema", np_ema)

            # EMA var (debiased): float/np
            float_ema_var = csp.stats.ema_var(test_data_float, halflife=timedelta(seconds=20), adjust=False)
            float_ema_var_adjusted = csp.stats.ema_var(test_data_float, halflife=timedelta(seconds=20), adjust=True)
            np_ema_var = csp.stats.ema_var(test_data_np, halflife=timedelta(seconds=20), adjust=False)
            csp.add_graph_output("float_ema_var", float_ema_var)
            csp.add_graph_output("np_ema_var", np_ema_var)
            csp.add_graph_output("float_ema_var_adjusted", float_ema_var_adjusted)

        res = csp.run(graph, starttime=times[0] - timedelta(seconds=1), endtime=times[-1], output_numpy=True)

        golden = (
            pl.Series(values=values)
            .ewm_mean_by(by=pl.Series(values=list(times)), half_life=timedelta(seconds=5))
            .to_pandas()
        )
        assert_series_equal(golden, pd.Series(res["float_ema"][1]), check_names=False)
        for element in range(N_ELEM_NP):
            assert_series_equal(golden, pd.Series(np.stack(res["np_ema"][1])[:, element]), check_names=False)

        # sanity check for variance (no open-source impl to compare to)
        with self.assertRaises(AssertionError):
            assert_series_equal(
                pd.Series(res["float_ema_var"][1]), pd.Series(res["float_ema_var_adjusted"][1]), check_names=False
            )
        for element in range(N_ELEM_NP):
            assert_series_equal(
                pd.Series(res["float_ema_var"][1]),
                pd.Series(np.stack(res["np_ema_var"][1])[:, element]),
                check_names=False,
            )

    def test_scalar_arrays(self):
        # np scalar arrays have no dimensions i.e. shape=(), but do contain a valid value
        def scalar_graph():
            raw_data = csp.count(csp.timer(timedelta(seconds=1), True))
            zero_dim_array_data = csp.apply(raw_data, lambda x: np.array(float(x)), np.ndarray)
            ema = csp.stats.ema(zero_dim_array_data, halflife=timedelta(seconds=10))
            sum = csp.stats.sum(zero_dim_array_data, interval=10, min_window=1)

            csp.add_graph_output("ema", ema)
            csp.add_graph_output("sum", sum)

        N_DATA_POINTS = 50
        res = csp.run(
            scalar_graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=N_DATA_POINTS), output_numpy=True
        )
        data = pd.Series(range(1, 51))

        self.assertTrue(res["ema"][1][0].shape == tuple())  # 0-dimension shape is preserved
        self.assertTrue(res["sum"][1][0].shape == tuple())  # 0-dimension shape is preserved
        assert_series_equal(
            pd.Series([res["ema"][1][k].item() for k in range(N_DATA_POINTS)]), data.ewm(halflife=10).mean()
        )
        assert_series_equal(
            pd.Series([res["sum"][1][k].item() for k in range(N_DATA_POINTS)]),
            data.rolling(window=10, min_periods=1).sum(),
        )


if __name__ == "__main__":
    unittest.main()
