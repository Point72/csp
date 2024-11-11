import math
import numpy as np
import numpy.testing
import pandas as pd
import sys
import unittest
from datetime import datetime, timedelta

import csp

# from csp.typing import Numpy1DArray, NumpyNDArray


class TestTA(unittest.TestCase):
    def test_macd(self):
        dvalues = np.random.uniform(low=-100, high=100, size=(1000,))
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(milliseconds=i + 1), dvalues[i]) for i in range(1000)])
            macd = csp.ta.macd(x, a=26, b=12)

            csp.add_graph_output("macd", macd)

        values = pd.Series(dvalues)
        pd_fast = values.ewm(span=12, min_periods=12).mean()
        pd_slow = values.ewm(span=26, min_periods=26).mean()
        pd_macd = pd_fast - pd_slow
        pd_macd = pd_macd.to_numpy().astype(float)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(milliseconds=1000))
        res = np.array(results["macd"])[:, 1].astype(float)

        np.testing.assert_allclose(res, pd_macd, rtol=1e-06)

    def test_bollinger(self):
        dvalues = np.random.uniform(low=-100, high=100, size=(1000,))
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(milliseconds=i + 1), dvalues[i]) for i in range(1000)])
            t = csp.ta.bollinger(x, 2, 20)
            upper, middle, lower = t["upper"], t["middle"], t["lower"]

            csp.add_graph_output("upper_bb", upper)
            csp.add_graph_output("middle_bb", middle)
            csp.add_graph_output("lower_bb", lower)

        values = pd.Series(dvalues)
        pd_middle = values.rolling(window=20).mean()
        pd_std = values.rolling(window=20).std()

        pd_upper = pd_middle + (pd_std * 2)
        pd_lower = pd_middle - (pd_std * 2)

        results = csp.run(graph, starttime=st, endtime=st + timedelta(milliseconds=1000))

        pd_upper = pd_upper.to_numpy().astype(float)
        pd_middle = pd_middle.to_numpy().astype(float)
        pd_lower = pd_lower.to_numpy().astype(float)

        upper = np.array(results["upper_bb"])[:, 1].astype(float)
        middle = np.array(results["middle_bb"])[:, 1].astype(float)
        lower = np.array(results["lower_bb"])[:, 1].astype(float)

        np.testing.assert_allclose(pd_upper, upper)
        np.testing.assert_allclose(pd_middle, middle)
        np.testing.assert_allclose(pd_lower, lower)

    def test_mom(self):
        dvalues = np.random.uniform(low=-100, high=100, size=(1000,))
        st = datetime(2020, 1, 1)
        n = 2

        @csp.graph
        def graph():
            x = csp.curve(typ=float, data=[(st + timedelta(milliseconds=i + 1), dvalues[i]) for i in range(1000)])
            mom = csp.ta.momentum(x, n)

            csp.add_graph_output("mom", mom)

        mom_n = [float("NaN")] * 1000

        for i in range(n, len(dvalues)):
            mom_n[i] = dvalues[i] - dvalues[i - n]

        res = csp.run(graph, starttime=st, endtime=st + timedelta(milliseconds=1000))
        mom = np.array(res["mom"])[:, 1].astype(float)
        np.testing.assert_allclose(mom_n, mom)

    def test_obv(self):
        close_values = np.random.uniform(low=-100, high=100, size=(1000,))
        volume_values = np.random.uniform(low=-100, high=100, size=(1000,))
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            close = csp.curve(
                typ=float, data=[(st + timedelta(milliseconds=i + 1), close_values[i]) for i in range(1000)]
            )
            volume = csp.curve(
                typ=float, data=[(st + timedelta(milliseconds=i + 1), volume_values[i]) for i in range(1000)]
            )

            obv = csp.ta.obv(close, volume)
            csp.add_graph_output("obv", obv)

        obv_values = np.zeros_like(close_values)
        for i in range(1, len(obv_values)):
            if close_values[i] > close_values[i - 1]:
                obv_values[i] = obv_values[i - 1] + volume_values[i]
            elif close_values[i] < close_values[i - 1]:
                obv_values[i] = obv_values[i - 1] - volume_values[i]
            else:
                obv_values[i] = obv_values[i - 1]

        res = csp.run(graph, starttime=st, endtime=st + timedelta(milliseconds=1000))
        obv = np.array(res["obv"])[:, 1].astype(float)
        np.testing.assert_allclose(obv_values, obv)
