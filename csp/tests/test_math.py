import math
import sys
import unittest
from datetime import datetime, timedelta

import numpy as np

import csp


class TestMath(unittest.TestCase):
    def test_math_binary_ops(self):
        OPS = {
            csp.add: lambda x, y: x + y,
            csp.sub: lambda x, y: x - y,
            csp.multiply: lambda x, y: x * y,
            csp.divide: lambda x, y: x / y,
            csp.pow: lambda x, y: x**y,
            csp.min: lambda x, y: min(x, y),
            csp.max: lambda x, y: max(x, y),
            csp.floordiv: lambda x, y: x // y,
        }

        @csp.graph
        def graph(use_promotion: bool):
            x = csp.count(csp.timer(timedelta(seconds=0.25)))
            if use_promotion:
                y = 10
                y_edge = csp.const(y)
            else:
                y = csp.default(csp.count(csp.timer(timedelta(seconds=1))), 1, delay=timedelta(seconds=0.25))
                y_edge = y

            csp.add_graph_output("x", csp.merge(x, csp.sample(y_edge, x)))
            csp.add_graph_output("y", csp.merge(y_edge, csp.sample(x, y_edge)))

            for op in OPS.keys():
                if use_promotion:
                    if op in [csp.min, csp.max]:
                        continue  # can't type promote, it's not being called ON an edge
                    p_op = OPS[op]
                    csp.add_graph_output(op.__name__, p_op(x, y))
                    csp.add_graph_output(op.__name__ + "-rev", p_op(y, x))
                else:
                    csp.add_graph_output(op.__name__, op(x, y))
                    csp.add_graph_output(op.__name__ + "-rev", op(y, x))

        for use_promotion in [False, True]:
            st = datetime(2020, 1, 1)
            results = csp.run(graph, use_promotion, starttime=st, endtime=st + timedelta(seconds=3))
            xv = [v[1] for v in results["x"]]
            yv = [v[1] for v in results["y"]]

            for op, comp in OPS.items():
                if op in [csp.min, csp.max] and use_promotion:
                    continue
                self.assertEqual(
                    [v[1] for v in results[op.__name__]], [comp(x, y) for x, y in zip(xv, yv)], op.__name__
                )
                self.assertEqual(
                    [v[1] for v in results[op.__name__ + "-rev"]], [comp(y, x) for x, y in zip(xv, yv)], op.__name__
                )

    def test_math_binary_ops_numpy(self):
        OPS = {
            csp.add: lambda x, y: x + y,
            csp.sub: lambda x, y: x - y,
            csp.multiply: lambda x, y: x * y,
            csp.divide: lambda x, y: x / y,
            csp.pow: lambda x, y: x**y,
            csp.min: lambda x, y: np.minimum(x, y),
            csp.max: lambda x, y: np.maximum(x, y),
            csp.floordiv: lambda x, y: x // y,
        }

        @csp.graph
        def graph(use_promotion: bool):
            x = csp.count(csp.timer(timedelta(seconds=0.25))) + csp.const(np.random.rand(10))
            if use_promotion:
                y = 10
                y_edge = csp.const(y)
            else:
                y = csp.default(
                    csp.count(csp.timer(timedelta(seconds=1))), 1, delay=timedelta(seconds=0.25)
                ) * csp.const(np.random.randint(1, 2, (10,)))
                y_edge = y

            csp.add_graph_output("x", csp.merge(x, csp.sample(y_edge, x)))
            csp.add_graph_output("y", csp.merge(y_edge, csp.sample(x, y_edge)))

            for op in OPS.keys():
                if use_promotion:
                    if op in [csp.min, csp.max]:
                        continue  # can't type promote, it's not being called ON an edge
                    p_op = OPS[op]
                    csp.add_graph_output(op.__name__, p_op(x, y))
                    csp.add_graph_output(op.__name__ + "-rev", p_op(y, x))
                else:
                    csp.add_graph_output(op.__name__, op(x, y))
                    csp.add_graph_output(op.__name__ + "-rev", op(y, x))

        for use_promotion in [False, True]:
            st = datetime(2020, 1, 1)
            results = csp.run(graph, use_promotion, starttime=st, endtime=st + timedelta(seconds=3))
            xv = [v[1] for v in results["x"]]
            yv = [v[1] for v in results["y"]]

            for op, comp in OPS.items():
                if op in [csp.min, csp.max] and use_promotion:
                    continue
                for i, (_, result) in enumerate(results[op.__name__]):
                    reference = comp(xv[i], yv[i])
                    self.assertTrue((result == reference).all(), op.__name__)
                for i, (_, result) in enumerate(results[op.__name__ + "-rev"]):
                    reference = comp(yv[i], xv[i])
                    self.assertTrue((result == reference).all(), op.__name__)

    def test_math_unary_ops(self):
        OPS = {
            csp.pos: lambda x: +x,
            csp.neg: lambda x: -x,
            csp.abs: lambda x: abs(x),
            csp.ln: lambda x: math.log(x),
            csp.log2: lambda x: math.log2(x),
            csp.log10: lambda x: math.log10(x),
            csp.exp: lambda x: math.exp(x),
            csp.exp2: lambda x: 2**x,
            csp.sin: lambda x: math.sin(x),
            csp.cos: lambda x: math.cos(x),
            csp.tan: lambda x: math.tan(x),
            csp.arctan: lambda x: math.atan(x),
            csp.sinh: lambda x: math.sinh(x),
            csp.cosh: lambda x: math.cosh(x),
            csp.tanh: lambda x: math.tanh(x),
            csp.arcsinh: lambda x: math.asinh(x),
            csp.arccosh: lambda x: math.acosh(x),
            csp.erf: lambda x: math.erf(x),
        }

        # Wheel builds on GH seem to produce slightly different results, maybe the python
        # version built against was compiled with slightly different math args?
        EPSILONS = {}
        if sys.platform == "win32":
            EPSILONS[csp.arcsinh] = 1e-12
            EPSILONS[csp.arccosh] = 1e-12

        @csp.graph
        def graph():
            x = csp.count(csp.timer(timedelta(seconds=0.25)))
            csp.add_graph_output("x", x)

            for op in OPS.keys():
                csp.add_graph_output(op.__name__, op(x))

        st = datetime(2020, 1, 1)
        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=3))
        xv = [v[1] for v in results["x"]]

        for op, comp in OPS.items():
            eps = EPSILONS.get(op, None)
            if eps:
                for v, x in zip(results[op.__name__], xv):
                    self.assertAlmostEqual(v[1], comp(x), msg=op.__name__, delta=eps)
            else:
                self.assertEqual([v[1] for v in results[op.__name__]], [comp(x) for x in xv], op.__name__)

    def test_math_unary_ops_numpy(self):
        OPS = {
            csp.abs: lambda x: np.abs(x),
            csp.ln: lambda x: np.log(x),
            csp.log2: lambda x: np.log2(x),
            csp.log10: lambda x: np.log10(x),
            csp.exp: lambda x: np.exp(x),
            csp.exp2: lambda x: np.exp2(x),
            csp.sin: lambda x: np.sin(x),
            csp.cos: lambda x: np.cos(x),
            csp.tan: lambda x: np.tan(x),
            csp.arctan: lambda x: np.arctan(x),
            csp.sinh: lambda x: np.sinh(x),
            csp.cosh: lambda x: np.cosh(x),
            csp.tanh: lambda x: np.tanh(x),
            csp.arcsinh: lambda x: np.arcsinh(x),
            csp.arccosh: lambda x: np.arccosh(x),
            # csp.erf: lambda x: math.erf(x),
        }

        @csp.graph
        def graph():
            x = csp.count(csp.timer(timedelta(seconds=0.25))) + csp.const(np.random.rand(10))
            csp.add_graph_output("x", x)

            for op in OPS.keys():
                csp.add_graph_output(op.__name__, op(x))

        st = datetime(2020, 1, 1)
        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=3))
        xv = [v[1] for v in results["x"]]

        for op, comp in OPS.items():
            for i, (_, result) in enumerate(results[op.__name__]):
                reference = comp(xv[i])
                # drop nans
                result = result[~np.isnan(result)]
                reference = reference[~np.isnan(reference)]
                self.assertTrue((result == reference).all(), op.__name__)

    def test_math_unary_ops_other_domain(self):
        OPS = {
            csp.arcsin: lambda x: math.asin(x),
            csp.arccos: lambda x: math.acos(x),
            csp.arctanh: lambda x: math.atanh(x),
        }

        EPSILONS = {}
        if sys.platform == "win32":
            EPSILONS[csp.arctanh] = 1e-12

        @csp.graph
        def graph():
            x = 1 / (csp.count(csp.timer(timedelta(seconds=0.25))) * math.pi)
            csp.add_graph_output("x", x)

            for op in OPS.keys():
                csp.add_graph_output(op.__name__, op(x))

        st = datetime(2020, 1, 1)
        results = csp.run(graph, starttime=st, endtime=st + timedelta(seconds=3))
        xv = [v[1] for v in results["x"]]

        for op, comp in OPS.items():
            eps = EPSILONS.get(op, None)
            if eps:
                for v, x in zip(results[op.__name__], xv):
                    self.assertAlmostEqual(v[1], comp(x), msg=op.__name__, delta=eps)
            else:
                self.assertEqual([v[1] for v in results[op.__name__]], [comp(x) for x in xv], op.__name__)

    def test_comparisons(self):
        OPS = {
            csp.gt: lambda x, y: x > y,
            csp.ge: lambda x, y: x >= y,
            csp.lt: lambda x, y: x < y,
            csp.le: lambda x, y: x <= y,
            csp.eq: lambda x, y: x == y,
            csp.ne: lambda x, y: x != y,
        }

        @csp.graph
        def graph(use_promotion: bool):
            x = csp.count(csp.timer(timedelta(seconds=0.25)))
            if use_promotion:
                y = 10
                y_edge = csp.const(y)
            else:
                y = csp.default(csp.count(csp.timer(timedelta(seconds=1))), 1, delay=timedelta(seconds=0.25))
                y_edge = y

            csp.add_graph_output("x", csp.merge(x, csp.sample(y_edge, x)))
            csp.add_graph_output("y", csp.merge(y_edge, csp.sample(x, y_edge)))

            for op in OPS.keys():
                if use_promotion:
                    p_op = OPS[op]
                    csp.add_graph_output(op.__name__, p_op(x, y))
                    csp.add_graph_output(op.__name__ + "-rev", p_op(y, x))
                else:
                    csp.add_graph_output(op.__name__, op(x, y))
                    csp.add_graph_output(op.__name__ + "-rev", op(y, x))

        for use_promotion in [False, True]:
            st = datetime(2020, 1, 1)
            results = csp.run(graph, use_promotion, starttime=st, endtime=st + timedelta(seconds=10))
            xv = [v[1] for v in results["x"]]
            yv = [v[1] for v in results["y"]]

            for op, comp in OPS.items():
                self.assertEqual(
                    [v[1] for v in results[op.__name__]], [comp(x, y) for x, y in zip(xv, yv)], op.__name__
                )
                self.assertEqual(
                    [v[1] for v in results[op.__name__ + "-rev"]], [comp(y, x) for x, y in zip(xv, yv)], op.__name__
                )

    def test_boolean_ops(self):
        def graph():
            x = csp.default(csp.curve(bool, [(timedelta(seconds=s), s % 2 == 0) for s in range(1, 20)]), False)
            y = csp.default(csp.curve(bool, [(timedelta(seconds=s * 0.5), s % 2 == 0) for s in range(1, 40)]), False)
            z = csp.default(csp.curve(bool, [(timedelta(seconds=s * 2), s % 2 == 0) for s in range(1, 10)]), False)

            csp.add_graph_output("rawx", x)
            csp.add_graph_output("x", csp.merge(x, csp.merge(csp.sample(y, x), csp.sample(z, x))))
            csp.add_graph_output("y", csp.merge(y, csp.merge(csp.sample(x, y), csp.sample(z, y))))
            csp.add_graph_output("z", csp.merge(z, csp.merge(csp.sample(x, z), csp.sample(y, z))))

            csp.add_graph_output("and_", csp.and_(x, y, z))
            csp.add_graph_output("or_", csp.or_(x, y, z))
            csp.add_graph_output("not_", csp.not_(x))

        results = csp.run(graph, starttime=datetime(2020, 5, 18))
        x = [v[1] for v in results["x"]]
        y = [v[1] for v in results["y"]]
        z = [v[1] for v in results["z"]]

        self.assertEqual([v[1] for v in results["and_"]], [all([a, b, c]) for a, b, c in zip(x, y, z)])
        self.assertEqual([v[1] for v in results["or_"]], [any([a, b, c]) for a, b, c in zip(x, y, z)])
        self.assertEqual([v[1] for v in results["not_"]], [not v[1] for v in results["rawx"]])
        pass


if __name__ == "__main__":
    unittest.main()
