import copy
import sys
import unittest

import csp
from csp import ts
from csp.impl.mem_cache import CspGraphObjectsMemCache, csp_memoized
from csp.impl.wiring.runtime import build_graph


class TestGraphMemCache(unittest.TestCase):
    def test_basic_functionality(self):
        with CspGraphObjectsMemCache.new_context():
            default_object = object()
            default_object2 = object()

            @csp_memoized
            def provider(arg1, arg2=None, arg3=None, arg4=None):
                return (arg1, arg2, arg3, arg4)

            o1 = provider(default_object)
            o2 = provider(arg1=o1[0])
            self.assertIs(o1, o2)
            o3 = provider(object(), arg3=object())
            self.assertIsNot(o1, o3)
            self.assertTrue(o3[0] is not None and o3[1] is None and o3[2] is not None and o3[3] is None)
            o4 = provider(object(), object(), object(), object())
            o5 = provider(*o4)
            o6 = provider(**dict(zip(("arg1", "arg2", "arg3", "arg4"), o5)))
            self.assertIs(o4, o5)
            self.assertIs(o5, o6)

            tuple_outside_context = provider(1)

            with CspGraphObjectsMemCache.new_context() as c:
                orig_context = c
                tuple_within_context2 = provider(1)
            # Anything that is defined outside of context is retrieved within context so it's the same object
            self.assertIs(tuple_outside_context, tuple_within_context2)

    def test_container_arguments(self):
        @csp_memoized
        def wrap_in_tuple(value):
            return (value,)

        @csp_memoized
        def new_object(value):
            return object()

        with CspGraphObjectsMemCache.new_context():
            o1 = wrap_in_tuple({"a", "b", "c"})
            o2 = wrap_in_tuple({"c", "b", "a"})
            self.assertIs(o1, o2)
            self.assertIsInstance(o1[0], set)
            o3 = wrap_in_tuple({"a": 1, "b": 2})
            o4 = wrap_in_tuple({"b": 2, "a": 1})
            o5 = wrap_in_tuple({"a": 3.0, "b": 1})
            o6 = wrap_in_tuple({"c": 2, "b": 1})
            self.assertIs(o3, o4)
            self.assertIsNot(o3, o5)
            self.assertIsNot(o3, o6)
            self.assertIsInstance(o3[0], dict)
            self.assertIsInstance(o4[0], dict)
            self.assertIsInstance(o5[0], dict)
            self.assertIsInstance(o6[0], dict)

            tuples = [(1, 2, 3), (1, 2, 3), (3, 2, 1)]

            complex_key = {"a": 1, "b": tuples, "c": [o1, o2, o3, o4, o5, o6], "d": "lists"}
            complex_key_copy = copy.copy(complex_key)
            self.assertIsNot(complex_key, complex_key_copy)
            self.assertIs(new_object(complex_key), new_object(complex_key))
            self.assertIs(new_object(complex_key), new_object(complex_key_copy))
            complex_key.pop("a")
            self.assertIsNot(new_object(complex_key), new_object(complex_key_copy))

    def _subgraph_aux(self, memoize=True, force_memoize=False):
        mutable_count = [0]

        node_decor = (
            csp.node if memoize and not force_memoize else csp.node(None, memoize=False, force_memoize=force_memoize)
        )
        graph_decor = (
            csp.graph if memoize and not force_memoize else csp.graph(None, memoize=False, force_memoize=force_memoize)
        )

        @node_decor
        def f(
            inp_ts: csp.ts["T"], inp_basket: [csp.ts["T"]], inp_basket2: {str: csp.ts["T"]}, scalar: object
        ) -> csp.ts["T"]:
            if csp.ticked(inp_ts):
                return inp_ts

        @graph_decor
        def sub_graph() -> csp.Outputs(e1=ts[int], e2=ts[int], e3=ts[int], e4=ts[float]):
            mutable_count[0] += 1
            c1 = csp.const(1)
            c2 = csp.const(1)
            c3 = csp.const(1.0)
            c4 = csp.const(5.0)
            e1 = f(c1, [c1, c2], {"c1": c1, "c2": c2}, None)
            e2 = f(c1, [c1, c2], {"c1": c1, "c2": c2}, None)
            e3 = f(c2, [c1, c1], {"c1": c1, "c2": c1}, None)
            e4 = f(c1, [c1, c3], {"c1": c1, "c2": c2}, None)
            return csp.output(e1=e1, e2=e2, e3=e3, e4=e4)

        return mutable_count, sub_graph

    def _check_memoized(self, mutable_count, sub_graph, all_memoize_disabled=False):
        def graph():
            g1 = sub_graph()
            self.assertEqual(mutable_count[0], 1)

            self.assertIs(g1.e1, g1.e2)
            if all_memoize_disabled:
                self.assertIsNot(g1.e1, g1.e3)
            else:
                self.assertIs(g1.e1, g1.e3)
            self.assertIsNot(g1.e1, g1.e4)
            if all_memoize_disabled:
                self.assertIsNot(g1.e1 + g1.e2, g1.e1 + g1.e2)
            else:
                # The "+" is still mem cached
                self.assertIs(g1.e1 + g1.e2, g1.e1 + g1.e2)

            g2 = sub_graph()
            self.assertEqual(mutable_count[0], 1)
            self.assertIs(g1, g2)

        build_graph(graph)

    def _check_no_memoization(self, mutable_count, sub_graph, all_memoize_disabled=False):
        def graph():
            g1 = sub_graph()
            self.assertEqual(mutable_count[0], 1)

            self.assertIsNot(g1.e1, g1.e2)
            self.assertIsNot(g1.e1, g1.e3)
            self.assertIsNot(g1.e1, g1.e4)
            if all_memoize_disabled:
                self.assertIsNot(g1.e1 + g1.e2, g1.e1 + g1.e2)
            else:
                # The "+" is still mem cached
                self.assertIs(g1.e1 + g1.e2, g1.e1 + g1.e2)

            g2 = sub_graph()
            self.assertEqual(mutable_count[0], 2)
            self.assertIsNot(g1, g2)

        build_graph(graph)

    def test_graph(self):
        self._check_memoized(*self._subgraph_aux())

    def test_graph_no_caching(self):
        self._check_no_memoization(*self._subgraph_aux(False))

    def test_memoize_disable_enabled(self):
        with csp.memoize(False):
            self._check_no_memoization(
                *self._subgraph_aux(memoize=True, force_memoize=False), all_memoize_disabled=True
            )
            self._check_memoized(*self._subgraph_aux(memoize=True, force_memoize=True), all_memoize_disabled=True)
            with csp.memoize(True):
                self._check_memoized(*self._subgraph_aux(memoize=True, force_memoize=False), all_memoize_disabled=False)
                self._check_memoized(*self._subgraph_aux(memoize=False, force_memoize=True), all_memoize_disabled=False)
                self._check_no_memoization(
                    *self._subgraph_aux(memoize=False, force_memoize=False), all_memoize_disabled=False
                )

            self._check_no_memoization(
                *self._subgraph_aux(memoize=True, force_memoize=False), all_memoize_disabled=True
            )
            self._check_memoized(*self._subgraph_aux(memoize=True, force_memoize=True), all_memoize_disabled=True)
            self._check_memoized(*self._subgraph_aux(memoize=False, force_memoize=True), all_memoize_disabled=True)
        self._check_memoized(*self._subgraph_aux(memoize=True, force_memoize=False), all_memoize_disabled=False)

    def test_pure_memoize(self):
        with CspGraphObjectsMemCache.new_context():
            count = [0]

            @csp.csp_memoized
            def f(val):
                count[0] += 1
                return val + count[0]

            self.assertEqual(f(1), 2)
            self.assertEqual(f(1), 2)
            with csp.memoize(False):
                self.assertEqual(f(1), 3)
            self.assertEqual(f(1), 2)

            class A:
                def __init__(self):
                    self._value = 0

                @csp.csp_memoized
                def f(self):
                    self._value += 1
                    return self._value

            a = A()
            self.assertEqual(a.f(), 1)
            self.assertEqual(a.f(), 1)
            with csp.memoize(False):
                self.assertEqual(a.f(), 2)
                self.assertEqual(a.f(), 3)
            self.assertEqual(a.f(), 1)


if __name__ == "__main__":
    unittest.main()
