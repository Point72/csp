import unittest
from datetime import datetime, timedelta

import numpy as np

import csp
from csp.random import brownian_motion, brownian_motion_1d, poisson_timer
from csp.typing import Numpy1DArray, NumpyNDArray


class TestPoissonTimer(unittest.TestCase):
    def test_simple(self):
        rate = csp.const(2.0)  # two events per second
        out = csp.run(
            poisson_timer, rate, seed=1234, value="foo", starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1000)
        )[0]
        # Should be about 2k events, subject to random noise
        self.assertGreater(len(out), 1900)
        self.assertLess(len(out), 2100)
        self.assertEqual(out[0][1], "foo")

    def test_delayed_start(self):
        rate = csp.const(0.1, delay=timedelta(days=1))  # two events per second
        out = csp.run(
            poisson_timer, rate, seed=1234, value="foo", starttime=datetime(2020, 1, 1), endtime=timedelta(days=2)
        )[0]
        self.assertGreater(out[0][0], datetime(2020, 1, 2))

    def test_changing_rate(self):
        rate1 = 1.0
        rate3 = 0.1
        rate = csp.curve(
            float, [(datetime(2020, 1, 1, 0), rate1), (datetime(2020, 1, 1, 1), 0.0), (datetime(2020, 1, 1, 2), rate3)]
        )
        out = csp.run(poisson_timer, rate, seed=1234, starttime=datetime(2020, 1, 1), endtime=datetime(2020, 1, 1, 3))[
            0
        ]
        first_hour = [v for t, v in out if t <= datetime(2020, 1, 1, 1)]
        second_hour = [v for t, v in out if datetime(2020, 1, 1, 1) < t <= datetime(2020, 1, 1, 2)]
        third_hour = [v for t, v in out if t > datetime(2020, 1, 1, 2)]
        self.assertGreater(len(first_hour), 60 * 60 * rate1 * 0.9)
        self.assertLess(len(first_hour), 60 * 60 * rate1 * 1.1)
        self.assertEqual(len(second_hour), 0)
        self.assertGreater(len(third_hour), 60 * 60 * rate3 * 0.9)
        self.assertLess(len(third_hour), 60 * 60 * rate3 * 1.1)


class TestBrownianMotion(unittest.TestCase):
    def test_simple_increments(self):
        mean = np.array([10.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        # Use two second increment to make sure time scaling is ok
        n_seconds = 2
        trigger = csp.timer(timedelta(seconds=n_seconds))
        out = csp.run(
            brownian_motion,
            trigger,
            csp.const(mean),
            csp.const(cov),
            seed=1234,
            return_increments=True,
            starttime=datetime(2020, 1, 1),
            endtime=timedelta(seconds=2000),
        )[0]
        self.assertEqual(len(out), 1000)
        data = np.vstack([o[1] for o in out])
        err_mean = data.mean(axis=0) - mean * n_seconds
        err_cov = np.cov(data.T) - cov * n_seconds
        self.assertLess(np.abs(err_mean).max(), 0.2)
        self.assertLess(np.abs(err_cov).max(), 0.2)

    def test_bad_covariance(self):
        mean = np.array([10.0, 0.0])
        covs = []
        covs.append(np.array([[1.0], [2.0]]))  # Not square
        covs.append(np.array([[1.0]]))  # Not same length as drift
        covs.append(np.array([[1.0, 0.5], [1.0, 2.0]]))  # Not symmetric
        covs.append(np.array([[1.0, 10.0], [10.0, 2.0]]))  # Not positive semi-definite
        trigger = csp.timer(timedelta(seconds=1))
        for cov in covs:
            self.assertRaises(
                ValueError,
                csp.run,
                brownian_motion,
                trigger,
                csp.const(mean),
                csp.const(cov),
                seed=1234,
                return_increments=True,
                starttime=datetime(2020, 1, 1),
                endtime=timedelta(seconds=2),
            )

    def test_brownian_motion(self):
        # Test that increments add to brownian motion
        mean = np.array([10.0, 0.0])
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        trigger = csp.timer(timedelta(seconds=1))
        out = csp.run(
            brownian_motion,
            trigger,
            csp.const(mean),
            csp.const(cov),
            seed=1234,
            return_increments=True,
            starttime=datetime(2020, 1, 1),
            endtime=timedelta(seconds=100),
        )[0]
        data = np.vstack([o[1] for o in out])
        bm_out = csp.run(
            brownian_motion,
            trigger,
            csp.const(mean),
            csp.const(cov),
            seed=1234,
            starttime=datetime(2020, 1, 1),
            endtime=timedelta(seconds=100),
        )[0]
        err = bm_out[-1][1] - data.sum(axis=0)
        self.assertAlmostEqual(np.abs(err).max(), 0.0)

    def test_brownian_motion_1d(self):
        mean = 10.0
        cov = 1.0
        trigger = csp.timer(timedelta(seconds=1))
        out = csp.run(
            brownian_motion,
            trigger,
            csp.const(np.array([mean])),
            csp.const(np.array([[cov]])),
            seed=1234,
            starttime=datetime(2020, 1, 1),
            endtime=timedelta(seconds=100),
        )[0]
        out1 = csp.run(
            brownian_motion_1d,
            trigger,
            csp.const(mean),
            csp.const(cov),
            seed=1234,
            starttime=datetime(2020, 1, 1),
            endtime=timedelta(seconds=100),
        )[0]
        self.assertEqual(out[-1][1][0], out1[-1][1])

    def test_change_at_trigger(self):
        mean1 = np.array([1.0])
        mean2 = np.array([-1.0])
        cov = np.array([[0.0]])
        trigger = csp.timer(timedelta(seconds=1))
        drift = csp.curve(Numpy1DArray[float], [(datetime(2020, 1, 1), mean1), (datetime(2020, 1, 1, 0, 0, 1), mean2)])
        cov = csp.const(cov)
        out = csp.run(
            brownian_motion,
            trigger,
            drift,
            cov,
            seed=1234,
            return_increments=True,
            starttime=datetime(2020, 1, 1),
            endtime=datetime(2020, 1, 1, 0, 0, 2),
        )[0]
        target = [(datetime(2020, 1, 1, 0, 0, 1), np.array([1.0])), (datetime(2020, 1, 1, 0, 0, 2), np.array([-1.0]))]
        np.testing.assert_equal(out, target)

    def test_change_between_triggers(self):
        mean1 = np.array([1.0])
        mean2 = np.array([2.0])
        cov = np.array([[0.0]])
        trigger = csp.timer(timedelta(seconds=2))
        drift = csp.curve(Numpy1DArray[float], [(datetime(2020, 1, 1), mean1), (datetime(2020, 1, 1, 0, 0, 1), mean2)])
        cov = csp.const(cov)
        out = csp.run(
            brownian_motion,
            trigger,
            drift,
            cov,
            seed=1234,
            return_increments=True,
            starttime=datetime(2020, 1, 1),
            endtime=datetime(2020, 1, 1, 0, 0, 2),
        )[0]
        target = [(datetime(2020, 1, 1, 0, 0, 2), np.array([3.0]))]
        np.testing.assert_equal(out, target)

    def test_changing_parameters(self):
        mean1 = np.array([10.0, 0.0])
        mean2 = np.array([1.0, 0.0])
        cov1 = np.array([[1.0, 0.5], [0.5, 2.0]])
        cov2 = np.array([[0.0, 0.0], [0.0, 1.0]])  # Note zero covariance for first dim!
        n_seconds = 1
        trigger = csp.timer(timedelta(seconds=n_seconds))
        drift = csp.curve(Numpy1DArray[float], [(datetime(2020, 1, 1, 0), mean1), (datetime(2020, 1, 1, 1), mean2)])
        cov = csp.curve(NumpyNDArray[float], [(datetime(2020, 1, 1, 0), cov1), (datetime(2020, 1, 1, 1), cov2)])
        out = csp.run(
            brownian_motion,
            trigger,
            drift,
            cov,
            seed=1234,
            return_increments=True,
            starttime=datetime(2020, 1, 1),
            endtime=datetime(2020, 1, 1, 2),
        )[0]
        data1 = np.vstack([v for t, v in out if t <= datetime(2020, 1, 1, 1)])
        data2 = np.vstack([v for t, v in out if datetime(2020, 1, 1, 1) < t <= datetime(2020, 1, 1, 2)])

        err_mean1 = data1.mean(axis=0) - mean1 * n_seconds
        err_cov1 = np.cov(data1.T) - cov1 * n_seconds
        self.assertLess(np.abs(err_mean1).max(), 0.05)
        self.assertLess(np.abs(err_cov1).max(), 0.05)

        err_mean2 = data2.mean(axis=0) - mean2 * n_seconds
        err_cov2 = np.cov(data2.T) - cov2 * n_seconds
        self.assertLess(np.abs(err_mean2).max(), 0.05)
        self.assertLess(np.abs(err_cov2).max(), 0.05)

    def disable_test_performance(self):
        dim = 1_000
        N = 100_000
        mean = np.zeros(dim)
        cov = np.diag(np.ones(dim))
        trigger = csp.timer(timedelta(seconds=1))

        def graph():
            out = brownian_motion(trigger, csp.const(mean), csp.const(cov), seed=1234)
            csp.add_graph_output("BM", out, tick_count=1)

        start = datetime.utcnow()

        from threadpoolctl import threadpool_limits  # May need to pip install separately

        with threadpool_limits(limits=1):  # To limit numpy parallelism
            csp.run(
                graph,
                starttime=datetime(2020, 1, 1),
                endtime=timedelta(seconds=N),
            )
        end = datetime.utcnow()
        print(f"Elapsed: {end - start}")
