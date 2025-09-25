import collections
import gc
import os
import pickle
import random
import re
import sys
import time
import traceback
import typing
import unittest
from datetime import datetime, timedelta
from typing import Callable, Dict, List

import numpy as np
import psutil
import pytest

import csp
from csp import PushMode, ts
from csp.impl.types.instantiation_type_resolver import ArgTypeMismatchError, TSArgTypeMismatchError
from csp.impl.wiring.delayed_node import DelayedNodeWrapperDef
from csp.impl.wiring.runtime import build_graph
from csp.lib import _csptestlibimpl

USE_PYDANTIC = os.environ.get("CSP_PYDANTIC", True)


@csp.graph
def _dummy_graph():
    raise NotImplementedError()


@csp.node
def _dummy_node():
    raise NotImplementedError()


class TestEngine(unittest.TestCase):
    def test_simple(self):
        @csp.node
        def simple(x: ts[int]) -> ts[float]:
            if csp.ticked(x):
                return x / 2.0

        def graph():
            x = csp.const(5)
            y = simple(x)
            return y

        result = csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=datetime(2020, 2, 7, 9, 1))[0][0]
        self.assertEqual(result[0], datetime(2020, 2, 7, 9))
        self.assertEqual(result[1], 5 / 2.0)

    def test_multiple_inputs(self):
        def graph():
            x = csp.curve(int, [(timedelta(seconds=v + 1), v + 1) for v in range(4)])
            y = csp.curve(int, [(timedelta(seconds=(v + 1) * 2), v + 1) for v in range(2)])

            return csp.add(x, y)

        result = csp.run(graph, starttime=datetime(2020, 2, 7, 9))[0]
        self.assertEqual(
            result,
            [
                (datetime(2020, 2, 7, 9, 0, 2), 3),  # First tick only once x and y are valid
                (datetime(2020, 2, 7, 9, 0, 3), 4),
                (datetime(2020, 2, 7, 9, 0, 4), 6),
            ],
        )

    def test_state_noblock(self):
        def graph():
            @csp.node
            def state_no_block(x: ts[int]) -> ts[int]:
                with csp.state():
                    s_c = 0

                if csp.ticked(x):
                    s_c += x
                    return s_c

            return state_no_block(csp.curve(int, [(timedelta(seconds=v), v) for v in range(10)]))

            result = csp.run(graph, starttime=datetime(2020, 2, 7, 9))[0][-1]
            self.assertEqual(result, sum(range(10)))

    def test_state_withblock(self):
        def graph():
            @csp.node
            def state_with_block(x: ts[int], start: int) -> ts[int]:
                with csp.state():
                    s_c = start

                if csp.ticked(x):
                    s_c += x
                    return s_c

            x = csp.curve(int, [(timedelta(seconds=v), v) for v in range(10)])
            return state_with_block(x, 5)

            result = csp.run(graph, starttime=datetime(2020, 2, 7, 9))[0][-1]
            self.assertEqual(result, 5 + sum(range(10)))

    def test_stop_block(self):
        @csp.node
        def mutate_on_stop(x: ts[int], out: list):
            with csp.state():
                s_sum = 0
            with csp.stop():
                out.append(s_sum)

            if csp.ticked(x):
                s_sum += x

        out = []
        x = csp.timer(timedelta(seconds=1), 1)
        csp.run(mutate_on_stop, x, out, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))
        self.assertEqual(out[0], 10)

    def test_alarm(self):
        @csp.node
        def alarm_node(repetition: int, cancel: bool = False) -> ts[int]:
            with csp.alarms():
                alarm = csp.alarm(int)
            with csp.start():
                for x in range(repetition):
                    handle = csp.schedule_alarm(alarm, timedelta(seconds=1), x)
                    if cancel:
                        csp.cancel_alarm(alarm, handle)

            if csp.ticked(alarm):
                csp.schedule_alarm(alarm, timedelta(seconds=1), alarm + repetition)
                return alarm

        result = csp.run(alarm_node(1), starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual(len(result), 10)
        self.assertEqual([v[1] for v in result], list(range(10)))

        result = csp.run(alarm_node(2), starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual(len(result), 20)
        self.assertEqual([v[1] for v in result], list(range(20)))

        result = csp.run(alarm_node(1, True), starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual(len(result), 0)

        result = csp.run(alarm_node(2, True), starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual(len(result), 0)

        @csp.node
        def reschedule_node() -> ts[int]:
            with csp.alarms():
                main = csp.alarm(bool)
                rescheduled = csp.alarm(int)
            with csp.state():
                s_handle = None
                s_expected_time = None

            # TEst will reschedule the alarm twice before allowing it to trigger
            with csp.start():
                csp.schedule_alarm(main, timedelta(seconds=1), True)
                s_handle = csp.schedule_alarm(rescheduled, timedelta(seconds=10), 123)
                s_expected_time = csp.now() + timedelta(seconds=5)

            if csp.ticked(main):
                if csp.num_ticks(main) == 1:
                    csp.schedule_alarm(main, timedelta(seconds=1), True)
                    # closer execution, start + 10 -> start + (1+2)
                    s_handle = csp.reschedule_alarm(rescheduled, s_handle, timedelta(seconds=2))
                else:
                    # further execeution, start + (1+2) -> start + ( 2 + 3)
                    s_handle = csp.reschedule_alarm(rescheduled, s_handle, timedelta(seconds=3))

            if csp.ticked(rescheduled):
                self.assertEqual(csp.now(), s_expected_time)
                csp.stop_engine()

                # verify exception
                with self.assertRaisesRegex(ValueError, "attempting to reschedule expired handle"):
                    csp.reschedule_alarm(rescheduled, s_handle, timedelta(seconds=1))

                return rescheduled

        result = csp.run(reschedule_node, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=120))[0]
        self.assertEqual(result[0][1], 123)

        # Test for bug "Cant cancel/reschedule non-collapsed alarms"
        @csp.node
        def node() -> ts[int]:
            with csp.alarms():
                main = csp.alarm(int)
            with csp.state():
                s_handle = None

            with csp.start():
                for i in range(10):
                    h = csp.schedule_alarm(main, timedelta(), i)
                    if i == 5:
                        s_handle = h

            if csp.ticked(main):
                self.assertNotEqual(main, 5)
                # By 3, 5 should be deferred
                if main == 3:
                    csp.cancel_alarm(main, s_handle)

                return 1

        csp.run(node, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=1))

    def test_active_passive(self):
        @csp.node
        def active_passive(x: ts[int], y: ts[int]) -> ts[int]:
            if csp.ticked(y):
                if csp.num_ticks(y) % 2 == 1:
                    # intentionally testing multiple calls
                    csp.make_passive(x)
                    csp.make_passive(x)
                    csp.make_passive(x)
                else:
                    csp.make_active(x)
                    csp.make_active(x)

            if csp.ticked(x):
                return x

        x = csp.count(csp.timer(timedelta(seconds=1), True))
        y = csp.count(csp.timer(timedelta(seconds=1.01), True))
        result = csp.run(active_passive(x, y), starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual(
            result,
            [
                (datetime(2020, 2, 7, 9, 0, 1), 1),
                (datetime(2020, 2, 7, 9, 0, 3), 3),
                (datetime(2020, 2, 7, 9, 0, 5), 5),
                (datetime(2020, 2, 7, 9, 0, 7), 7),
                (datetime(2020, 2, 7, 9, 0, 9), 9),
            ],
        )

        @csp.node
        def active_passive(x: ts[int]) -> csp.Outputs(x=ts[int], active=ts[bool]):
            with csp.alarms():
                alarm = csp.alarm(bool)
            with csp.start():
                csp.schedule_alarm(alarm, timedelta(seconds=1), True)

            if csp.ticked(alarm):
                # Skew to get to 0
                if random.random() > 0.7:
                    csp.make_active(x)
                    csp.output(active=True)
                else:
                    csp.make_passive(x)
                    csp.output(active=False)

                csp.schedule_alarm(alarm, timedelta(seconds=1), True)

            if csp.ticked(x):
                csp.output(x=x)

        @csp.node
        def checkActiveTick(x_orig: ts[int], x: ts[int], active: ts[bool]):
            if csp.ticked(x_orig) and csp.valid(active):
                self.assertEqual(active, csp.ticked(x), (csp.now(), x_orig))

        def g():
            # intentionally misalign the input tick, we dont want it to tick at the same time as the alarm switch for this test
            x = csp.count(csp.timer(timedelta(seconds=1.0 / 3.0)))

            with csp.memoize(False):
                # Create a duplicate input so that we dont keep x active in checkActiveTick
                # we want to force consumers to go to and from 0
                x_dup = csp.count(csp.timer(timedelta(seconds=1.0 / 3.0)))
                # We want to exercise the consumer vector empty/single/vector logic
                # Ensure we create multiple consumers by turning off memoization
                for i in range(5):
                    res = active_passive(x)
                    checkActiveTick(x_dup, res.x, res.active)

        seed = int(time.time())
        print("USING SEED", seed)
        random.seed(seed)
        csp.run(g, starttime=datetime(2022, 7, 26), endtime=timedelta(minutes=30))

    def test_node_csp_count(self):
        @csp.node
        def count(x: ts[int]) -> ts[int]:
            if csp.ticked(x):
                return csp.num_ticks(x)

        x = csp.count(csp.timer(timedelta(seconds=1), True))
        result = csp.run(x, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual([v[1] for v in result], list(range(1, 11)))

    def test_named_outputs(self):
        @csp.node
        def split(x: ts[int]) -> csp.Outputs(even=ts[int], odd=ts[int]):
            if csp.ticked(x):
                if x % 2 == 0:
                    csp.output(even=x)
                else:
                    csp.output(odd=x)

        x = csp.count(csp.timer(timedelta(seconds=1), True))
        result = csp.run(split, x, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))
        self.assertEqual([v[1] for v in result["even"]], list(range(2, 11, 2)))
        self.assertEqual([v[1] for v in result["odd"]], list(range(1, 10, 2)))

    def test_named_return(self):
        @csp.node
        def split(x: ts[int]) -> csp.Outputs(even=ts[int], odd=ts[int]):
            if csp.ticked(x):
                if x % 2 == 0:
                    return csp.output(even=x)
                return csp.output(odd=x)

        x = csp.count(csp.timer(timedelta(seconds=1), True))
        result = csp.run(split, x, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))
        self.assertEqual([v[1] for v in result["even"]], list(range(2, 11, 2)))
        self.assertEqual([v[1] for v in result["odd"]], list(range(1, 10, 2)))

    def test_single_csp_output(self):
        @csp.node
        def count(x: ts[int]) -> ts[int]:
            if csp.ticked(x):
                csp.output(csp.num_ticks(x) * 2)

        x = csp.timer(timedelta(seconds=1), True)
        result = csp.run(count, x, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual([v[1] for v in result], list(x * 2 for x in range(1, 11)))

    def test_single_csp_numpy_output(self):
        @csp.node
        def count(x: ts[int]) -> ts[int]:
            if csp.ticked(x):
                csp.output(csp.num_ticks(x) * 2)

        x = csp.timer(timedelta(seconds=1), True)
        result = csp.run(count, x, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10), output_numpy=True)[
            0
        ]
        expected_times = np.array(
            list(np.datetime64(datetime(2020, 2, 7, 9) + timedelta(seconds=x)) for x in range(1, 11))
        )
        self.assertTrue(np.array_equal(result[0], expected_times))
        self.assertEqual(result[1].tolist(), list(x * 2 for x in range(1, 11)))

    def test_multi_csp_numpy_output(self):
        @csp.graph
        def my_graph():
            csp.add_graph_output("a", csp.const("foo"))
            csp.add_graph_output("b", csp.merge(csp.const(1), csp.const(2, timedelta(seconds=1))))

        result = csp.run(my_graph, starttime=datetime(2021, 6, 21), output_numpy=True)

        res1 = result["a"]
        expected_times_1 = np.array([np.datetime64(datetime(2021, 6, 21))])
        self.assertTrue(np.array_equal(res1[0], expected_times_1))
        self.assertEqual(res1[1].tolist(), ["foo"])

        res2 = result["b"]
        expected_times_2 = np.array([np.datetime64(datetime(2021, 6, 21) + timedelta(seconds=x)) for x in (0, 1)])
        self.assertTrue(np.array_equal(res2[0], expected_times_2))
        self.assertEqual(res2[1].tolist(), [1, 2])

    def test_single_valued_csp_numpy_output(self):
        # these are a special case where, due to optimization, there is no buffer so we only need the last value
        class Foo:
            pass

        foo = Foo()

        @csp.graph
        def g():
            csp.add_graph_output("a", csp.const("a"), tick_count=1)
            csp.add_graph_output("b", csp.count(csp.timer(timedelta(seconds=1), 1)), tick_count=1)
            csp.add_graph_output("c", csp.const(foo), tick_count=1)
            csp.add_graph_output("d", csp.const(datetime(2020, 1, 1)), tick_count=1)
            csp.add_graph_output("e", csp.const(timedelta(seconds=1)), tick_count=1)

        res = csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=2), output_numpy=True)

        # times
        exp_time_1 = np.array([np.datetime64(datetime(2020, 1, 1))])
        for out in ["a", "c", "d", "e"]:
            self.assertTrue(np.array_equal(res[out][0], exp_time_1))
        exp_time_2 = np.array([np.datetime64(datetime(2020, 1, 1) + timedelta(seconds=2))])
        self.assertTrue(np.array_equal(res["b"][0], exp_time_2))

        # values
        self.assertEqual(res["a"][1].tolist(), ["a"])
        self.assertEqual(res["b"][1].tolist(), [2])
        self.assertEqual(res["c"][1].tolist(), [foo])
        self.assertTrue(np.array_equal(res["d"][1], exp_time_1))
        self.assertTrue(np.array_equal(res["e"][1], np.array([np.timedelta64(timedelta(seconds=1))])))

    def test_single_csp_return(self):
        @csp.node
        def count(x: ts[int]) -> ts[int]:
            if csp.ticked(x):
                return csp.output(csp.num_ticks(x) * 2)

        x = csp.timer(timedelta(seconds=1), True)
        result = csp.run(count, x, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual([v[1] for v in result], list(x * 2 for x in range(1, 11)))

    def test_csp_now(self):
        @csp.node
        def times(x: ts[bool]) -> ts[datetime]:
            if csp.ticked(x):
                return csp.now()

        x = csp.timer(timedelta(seconds=1), True)
        result = csp.run(times(x), starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))[0]
        self.assertEqual([v[0] for v in result], [v[1] for v in result])

    def test_stop_engine(self):
        @csp.node
        def stop(x: ts[bool]) -> ts[bool]:
            if csp.ticked(x):
                if csp.num_ticks(x) == 5:
                    csp.stop_engine()
                return x

        result = csp.run(stop, csp.timer(timedelta(seconds=1)), starttime=datetime(2020, 5, 19))[0]
        self.assertEqual(len(result), 5)

    def test_tvar_validation_context_lifetime(self):
        import gc

        from csp.impl.types.pydantic_type_resolver import TVarValidationContext

        def count_contexts():
            return sum(1 for o in gc.get_objects() if type(o) is TVarValidationContext)

        @csp.node
        def echo(x: ts[int]) -> ts[int]:
            return x

        gc.collect(0)
        before = count_contexts()
        csp.build_graph(echo, realtime=False, x=csp.const(1))
        gc.collect(0)
        after = count_contexts()

        self.assertEqual(before, after)

    def test_class_member_node(self):
        class ClassWithNodes:
            def __init__(self):
                self._data = []

            @csp.node
            def member_node(self: object, x: ts[int]):
                """it is NOT recommended to mutate state in a node!!"""
                if csp.ticked(x):
                    self._data.append(x)

        c = ClassWithNodes()

        def graph():
            x = c.member_node(csp.count(csp.timer(timedelta(seconds=1), True)))

        csp.run(graph, starttime=datetime(2020, 2, 7, 9), endtime=timedelta(seconds=10))
        self.assertEqual(c._data, list(range(1, 11)))

    def test_duplicate_outputs(self):
        def graph():
            csp.add_graph_output(0, csp.const(1))
            return csp.const(2)

        with self.assertRaisesRegex(ValueError, 'graph output key "0" is already bound'):
            csp.run(graph, starttime=datetime.now())

    def test_with_support(self):
        # This test case tests a parsing bug that we had, where "with" statement at the main function block was causing parse error
        class ValueSetter(object):
            def __init__(self, l: List[int]):
                self._l = l

            def __enter__(self):
                self._l.append(1)

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._l.append(2)

        @csp.node
        def my_node(inp: ts[bool]) -> ts[List[int]]:
            with csp.state():
                l = []
            with ValueSetter(l):
                return l

        def graph():
            csp.add_graph_output("my_node", my_node(csp.timer(timedelta(seconds=1))))

        res = csp.run(graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=3))
        self.assertEqual(res["my_node"][-1][1], [1, 2, 1, 2, 1])
        self.assertEqual(res["my_node"][-2][1], [1, 2, 1])
        self.assertEqual(res["my_node"][-3][1], [1])

    def test_bugreport_csp28(self):
        """bug where non-basket inputs after basket inputs were not being assigne dproperly in c++"""

        @csp.node
        def buggy(basket: [ts[int]], x: ts[bool]) -> ts[bool]:
            if csp.ticked(x) and csp.valid(x):
                return x

        result = csp.run(buggy, [csp.const(1, delay=timedelta(seconds=1))], csp.const(True), starttime=datetime.now())[
            0
        ]
        self.assertEqual(len(result), 1)

    def test_output_validation(self):
        from csp.impl.wiring import CspParseError

        with self.assertRaisesRegex(CspParseError, "returning from node without any outputs defined"):

            @csp.node
            def n(x: ts[bool]):
                return 1

        with self.assertRaisesRegex(CspParseError, "returning from node without any outputs defined"):

            @csp.node
            def n(x: ts[bool]):
                csp.output(1)

        with self.assertRaisesRegex(CspParseError, "unrecognized output 'x'"):

            @csp.node
            def n(x: ts[bool]) -> ts[bool]:
                csp.output(x=1)

        with self.assertRaisesRegex(
            CspParseError, "node has __outputs__ defined but no return or csp.output statements"
        ):

            @csp.node
            def n(x: ts[bool]) -> ts[bool]:
                pass

        with self.assertRaisesRegex(CspParseError, "output 'y' is never returned"):

            @csp.node
            def n(x: ts[bool]) -> csp.Outputs(x=ts[bool], y=ts[bool]):
                csp.output(x=5)

        with self.assertRaisesRegex(CspParseError, "output 'y' is never returned"):

            @csp.node
            def n(x: ts[bool]) -> csp.Outputs(x=ts[bool], y=ts[bool]):
                return csp.output(x=5)

        with self.assertRaisesRegex(KeyError, "Output y is not returned from the graph"):

            @csp.graph
            def g() -> csp.Outputs(x=ts[int], y=ts[int]):
                return csp.output(x=csp.const(1), z=csp.const(3))

            csp.run(g, starttime=datetime.now(), endtime=timedelta(seconds=1))

    def test_access_before_valid(self):
        @csp.node
        def foo(x: ts[int], y: ts[int]):
            print(x + y)

        major, minor, *rest = sys.version_info
        if major >= 3 and minor >= 11:
            expected_str = "cannot access local variable"
        else:
            expected_str = "referenced before assignment"
        with self.assertRaisesRegex(UnboundLocalError, expected_str):
            csp.run(foo, csp.const(1), csp.const(1, delay=timedelta(1)), starttime=datetime.utcnow())

    def test_cell_access(self):
        '''was a bug "PyNode crashes on startup when cellvars are generated"'''

        # All types of ts inputs as cellvars, no args
        @csp.node
        def node(x: ts[int], y: [ts[int]], x2: ts[int], y2: [ts[int]], s: int) -> csp.Outputs(o1=ts[int], o2=ts[int]):
            with csp.state():
                s_1 = 1
                s_2 = 2

            ## Force as many combinations of locals vs cellvars as we can
            xl = lambda: x2
            yl = lambda: y2[0]
            ol = lambda v: csp.output(o2, v)
            sl = lambda: s_2
            # Node ref
            nl = lambda: csp.now()

            val = nl()
            out = xl() + yl() + sl()
            ol(out)

            csp.output(o1, 1)

        res = csp.run(
            node, csp.const(1), [csp.const(1)], csp.const(100), [csp.const(200)], 5, starttime=datetime.utcnow()
        )["o2"][0]
        self.assertEqual(res[1], 100 + 200 + 2)

        # Test arguments in cellvars werent being processed correctly

        # scalar only in cellvar
        @csp.node
        def node2(x: ts[int], s: int) -> ts[int]:
            f = lambda: s
            return x * f()

        res = csp.run(node2, csp.const(1), 5, starttime=datetime.utcnow())[0][0]
        self.assertEqual(res[1], 5)

        # scalar and ts in cellvar
        @csp.node
        def node3(x: ts[int], s: int) -> ts[int]:
            with csp.state():
                s_1 = 1
                s_2 = 2

            f = lambda: x * s
            return f()

        res = csp.run(node3, csp.const(1), 5, starttime=datetime.utcnow())[0][0]
        self.assertEqual(res[1], 5)

    def test_stop_time(self):
        '''was a bug "__stop__ csp.now() returns wrong time"'''

        @csp.node
        def t(x: ts[int], endtime: datetime):
            with csp.stop():
                self.assertEqual(csp.now(), endtime)

            if csp.ticked(x):
                pass

        st = datetime(2020, 6, 11)
        et = st + timedelta(seconds=10)
        csp.run(t, csp.timer(timedelta(seconds=1)), et, starttime=st, endtime=et)

        ## This checks that endtime aligns with time at time of a stop_engine call
        @csp.node
        def t(x: ts[int], endtime: datetime):
            with csp.alarms():
                stop = csp.alarm(bool)
            with csp.start():
                csp.schedule_alarm(stop, endtime, True)

            with csp.stop():
                self.assertEqual(csp.now(), endtime)

            if csp.ticked(stop):
                csp.stop_engine()

        st = datetime(2020, 6, 11)
        csp.run(t, csp.timer(timedelta(seconds=1)), st + timedelta(seconds=5), starttime=st, endtime=et)

    def test_duplicate_time(self):
        data = [
            (timedelta(seconds=0), 1),
            (timedelta(seconds=1), 2),
            (timedelta(seconds=1), 3),
            (timedelta(seconds=2), 4),
            (timedelta(seconds=2), 5),
            (timedelta(seconds=3), 6),
            (timedelta(seconds=4), 7),
        ]

        c = csp.curve(int, data, push_mode=csp.PushMode.NON_COLLAPSING)

        # Forcing through a node
        c = csp.filter(csp.const(True), c)
        st = datetime(2020, 1, 1)
        r = csp.run(c, starttime=st)[0]
        expected = [(st + d[0], d[1]) for d in data]
        self.assertEqual(r, expected)

        ## Test duplicate time on alarms
        @csp.node
        def alarms(data: list) -> ts[int]:
            with csp.alarms():
                tick = csp.alarm(int)
            with csp.state():
                s_index = 0
                s_start = csp.now()
            with csp.start():
                csp.schedule_alarm(tick, s_start + data[s_index][0], data[s_index][1])
                s_index += 1

            if csp.ticked(tick):
                csp.output(tick)
                if s_index < len(data):
                    csp.schedule_alarm(tick, s_start + data[s_index][0], data[s_index][1])
                    s_index += 1

        r = csp.run(alarms, data, starttime=st)[0]
        self.assertEqual(r, expected)

    def test_sim_push_mode(self):
        data = [
            (timedelta(seconds=0), 1),
            (timedelta(seconds=1), 2),
            (timedelta(seconds=1), 3),
            (timedelta(seconds=2), 4),
            (timedelta(seconds=2), 5),
            (timedelta(seconds=2), 6),
            (timedelta(seconds=3), 7),
            (timedelta(seconds=4), 8),
        ]

        def graph():
            lv = csp.curve(int, data, push_mode=csp.PushMode.LAST_VALUE)
            nc = csp.curve(int, data, push_mode=csp.PushMode.NON_COLLAPSING)
            b = csp.curve(int, data, push_mode=csp.PushMode.BURST)

            csp.add_graph_output("lv", lv)
            csp.add_graph_output("nc", nc)
            csp.add_graph_output("b", b)

        st = datetime(2020, 1, 1)
        results = csp.run(graph, starttime=st)

        self.assertEqual(results["nc"], [(st + td, v) for td, v in data])
        self.assertEqual(results["lv"], [(st + td, v) for td, v in data if v not in (2, 4, 5)])
        b = collections.defaultdict(list)
        for t, v in data:
            b[st + t].append(v)
        self.assertEqual(results["b"], list(b.items()))

    def test_managed_sim_input_pushmode(self):
        from csp.impl.adaptermanager import AdapterManagerImpl, ManagedSimInputAdapter
        from csp.impl.wiring import py_managed_adapter_def

        class TestAdapterManager:
            def __init__(self, data):
                self._data = data

            def subscribe(self, id, push_mode):
                return TestAdapter(self, int, id, push_mode=push_mode)

            def _create(self, engine, memo):
                return TestAdapterManagerImpl(engine, self)

        class TestAdapterManagerImpl(AdapterManagerImpl):
            def __init__(self, engine, adapterRep):
                super().__init__(engine)
                self._data = adapterRep._data
                self._inputs = {}
                self._idx = 0

            def start(self, starttime, endtime):
                pass

            def register_input_adapter(self, id, adapter):
                self._inputs[id] = adapter

            def process_next_sim_timeslice(self, now):
                if self._idx >= len(self._data):
                    return None

                while self._idx < len(self._data):
                    time, id, value = self._data[self._idx]
                    if time > now:
                        return time
                    self._inputs[id].push_tick(value)
                    self._idx += 1

                return None

        class TestAdapterImpl(ManagedSimInputAdapter):
            def __init__(self, managerImpl, typ, id):
                managerImpl.register_input_adapter(id, self)

        TestAdapter = py_managed_adapter_def(
            "test_adapter", TestAdapterImpl, ts["T"], TestAdapterManager, typ="T", id=str
        )

        st = datetime(2020, 6, 17)
        data = [
            (st + timedelta(seconds=1), "lv", 1),
            (st + timedelta(seconds=1), "lv", 2),
            (st + timedelta(seconds=1), "nc", 1),
            (st + timedelta(seconds=1), "nc", 2),
            (st + timedelta(seconds=1), "b", 1),
            (st + timedelta(seconds=1), "b", 2),
            (st + timedelta(seconds=2), "nc", 3),
            (st + timedelta(seconds=3), "lv", 3),
            (st + timedelta(seconds=4), "lv", 4),
            (st + timedelta(seconds=4), "lv", 5),
            (st + timedelta(seconds=4), "b", 3),
            (st + timedelta(seconds=4), "b", 4),
            (st + timedelta(seconds=4), "b", 5),
            (st + timedelta(seconds=4), "b", 6),
            (st + timedelta(seconds=5), "nc", 4),
            (st + timedelta(seconds=5), "nc", 5),
            (st + timedelta(seconds=5), "nc", 6),
            (st + timedelta(seconds=5), "b", 7),
            (st + timedelta(seconds=5), "b", 8),
            (st + timedelta(seconds=5), "b", 9),
        ]

        def graph():
            adapter = TestAdapterManager(data)

            nc = adapter.subscribe("nc", push_mode=csp.PushMode.NON_COLLAPSING)
            lv = adapter.subscribe("lv", push_mode=csp.PushMode.LAST_VALUE)
            b = adapter.subscribe("b", push_mode=csp.PushMode.BURST)

            csp.add_graph_output("nc", nc)
            csp.add_graph_output("lv", lv)
            csp.add_graph_output("b", b)

        results = csp.run(graph, starttime=st)
        self.assertEqual(results["lv"], [(v[0], v[2]) for v in data if v[1] == "lv" and v[2] not in (1, 4)])
        self.assertEqual(results["nc"], [(v[0], v[2]) for v in data if v[1] == "nc"])
        b = collections.defaultdict(list)
        for t, n, v in data:
            if n == "b":
                b[t].append(v)
        self.assertEqual(results["b"], list(b.items()))

    def test_adapter_manager_engine_shutdown(self):
        from csp.impl.adaptermanager import AdapterManagerImpl, ManagedSimInputAdapter
        from csp.impl.wiring import py_managed_adapter_def

        class TestAdapterManager:
            def __init__(self):
                self._impl = None

            def subscribe(self):
                return TestAdapter(self)

            def _create(self, engine, memo):
                self._impl = TestAdapterManagerImpl(engine)
                return self._impl

        class TestAdapterManagerImpl(AdapterManagerImpl):
            def __init__(self, engine):
                super().__init__(engine)

            def start(self, starttime, endtime):
                pass

            def stop(self):
                pass

            def process_next_sim_timeslice(self, now):
                try:
                    [].pop()
                except IndexError as e:
                    self.shutdown_engine(e)

        class TestAdapterImpl(ManagedSimInputAdapter):
            def __init__(self, manager_impl):
                pass

        TestAdapter = py_managed_adapter_def("TestAdapter", TestAdapterImpl, ts[int], TestAdapterManager)

        def graph():
            adapter = TestAdapterManager()
            nc = adapter.subscribe()
            csp.add_graph_output("nc", nc)

        try:
            csp.run(graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=1))
        except IndexError:
            tb = traceback.format_exc()

        self.assertTrue("[].pop()" in tb and "process_next_sim_timeslice" in tb)

    def test_feedback(self):
        # Dummy example
        class Request(csp.Struct):
            command: str

        class Reply(csp.Struct):
            response: str

        @csp.node
        def process_req(request: ts[Request]) -> ts[Reply]:
            with csp.alarms():
                reply = csp.alarm(Reply)
            if csp.ticked(request):
                csp.schedule_alarm(reply, timedelta(seconds=1), Reply(response="ack" + request.command))

            if csp.ticked(reply):
                return reply

        responses = []

        @csp.node
        def req_reply(command: ts[str], reply: ts[Reply]) -> ts[Request]:
            if csp.ticked(command):
                return Request(command=command)

            if csp.ticked(reply):
                responses.append((csp.now(), reply))

        @csp.graph
        def graph():
            commands = csp.curve(str, [(timedelta(seconds=x), str(x)) for x in range(10)])

            reply_fb = csp.feedback(Reply)
            requests = req_reply(commands, reply_fb.out())
            reply = process_req(requests)
            reply_fb.bind(reply)

            csp.add_graph_output("reply_fb", reply_fb.out())
            csp.add_graph_output("reply", reply)

        st = datetime(2020, 7, 7)
        results = csp.run(graph, starttime=st)
        self.assertEqual(responses, [(st + timedelta(seconds=x + 1), Reply(response="ack%s" % x)) for x in range(10)])
        self.assertEqual(responses, results["reply_fb"])
        self.assertEqual(responses, results["reply"])

        ## Test exceptions
        def graph():
            fb = csp.feedback(int)
            if USE_PYDANTIC:
                msg = ".*value passed to argument of type TsType must be an instance of Edge.*"
            else:
                msg = (
                    re.escape(r"""In function _bind: Expected csp.impl.types.tstype.TsType[""")
                    + ".*"
                    + re.escape(r"""('T')] for argument 'x', got 1 (int)""")
                )
            with self.assertRaisesRegex(TypeError, msg):
                fb.bind(1)

            if USE_PYDANTIC:
                msg = re.escape("cannot validate ts[str] as ts[int]: <class 'str'> is not a subclass of <class 'int'>")
            else:
                msg = re.escape(r"""In function _bind: Expected ts[T] for argument 'x', got ts[str](T=int)""")

            with self.assertRaisesRegex(TypeError, msg):
                fb.bind(csp.const("123"))

            fb.bind(csp.const(1))
            with self.assertRaisesRegex(RuntimeError, "csp.feedback is already bound"):
                fb.bind(csp.const(1))

        build_graph(graph)

        def unbound_graph():
            fb = csp.feedback(int)
            csp.print("test", fb.out())

        with self.assertRaisesRegex(RuntimeError, "unbound csp.feedback used in graph"):
            csp.run(unbound_graph, starttime=datetime.utcnow())

    def test_list_feedback_typecheck(self):
        @csp.graph
        def g() -> csp.ts[List[int]]:
            fb = csp.feedback(List[int])
            if USE_PYDANTIC:
                msg = re.escape(
                    "cannot validate ts[int] as ts[typing.List[int]]: <class 'int'> is not a subclass of <class 'list'>"
                )
            else:
                msg = re.escape(r"""Expected ts[T] for argument 'x', got ts[int](T=typing.List[int])""")
            with self.assertRaisesRegex(TypeError, msg):
                fb.bind(csp.const(42))

            fb.bind(csp.const([42]))
            return fb.out()

        res = csp.run(g, starttime=datetime.utcnow())
        self.assertEqual(res[0][0][1], [42])

        # Test Typing.List which was a bug "crash on feedback tick"
        @csp.graph
        def g() -> csp.ts[List[int]]:
            fb = csp.feedback(List[int])
            if USE_PYDANTIC:
                msg = re.escape(
                    "cannot validate ts[int] as ts[typing.List[int]]: <class 'int'> is not a subclass of <class 'list'>"
                )
            else:
                msg = re.escape(r"""Expected ts[T] for argument 'x', got ts[int](T=typing.List[int])""")
            with self.assertRaisesRegex(TypeError, msg):
                fb.bind(csp.const(42))

            fb.bind(csp.const([42]))
            return fb.out()

        res = csp.run(g, starttime=datetime.utcnow())
        self.assertEqual(res[0][0][1], [42])

    def test_list_inside_callable(self):
        '''was a bug "Empty list inside callable annotation raises exception"'''

        @csp.graph
        def graph(v: Dict[str, Callable[[], str]]):
            pass

        csp.run(graph, {"x": (lambda v: v)}, starttime=datetime(2020, 6, 17))

    def test_tuples_as_list(self):
        '''was a bug "Support tuples as list baskets"'''

        @csp.node
        def sum_vals(inputs: [ts["T"]]) -> ts["T"]:
            if csp.ticked(inputs) and csp.valid(inputs):
                return sum((inp for inp in inputs))

        @csp.graph
        def my_graph():
            my_ts = csp.timer(timedelta(hours=1), 1)
            tuple_basket = (my_ts, my_ts)

            csp.add_graph_output("sampled", sum_vals(tuple_basket))

        g = csp.run(my_graph, starttime=datetime(2020, 3, 1, 9, 30), endtime=timedelta(hours=0, minutes=390))

    def test_node_parse_stack(self):
        '''was a bug "Node parsing exception stacks are truncated, but type errors when invoking are not."'''

        @csp.node
        def aux(tag: str, my_arg: ts["T"]):
            pass

        @csp.node
        def f(x: ts[int]):
            __out__()
            pass

        def graph():
            x = f(csp.const(1))
            aux("x", x)

        try:
            build_graph(graph)
            # Should never get here
            self.assertFalse(True)
        except Exception as e:
            self.assertIsInstance(e, TypeError)
            traceback_list = list(
                filter(lambda v: v.startswith("File"), (map(str.strip, traceback.format_exc().split("\n"))))
            )
            self.assertTrue(__file__ in traceback_list[-1])
            self.assertLessEqual(len(traceback_list), 10)
            if USE_PYDANTIC:
                self.assertIn("value passed to argument of type TsType must be an instance of Edge", str(e))
            else:
                self.assertEqual(str(e), "In function aux: Expected ts[T] for argument 'my_arg', got None")

    def test_union_type_check(self):
        '''was a bug "Add support for typing.Union in type checking layer"'''

        @csp.graph
        def graph(x: typing.Union[int, float, str]):
            pass

        build_graph(graph, 1)
        build_graph(graph, 1.1)
        build_graph(graph, "s")
        if USE_PYDANTIC:
            # Pydantic's error reporting for unions is a bit quirky, as it reports a validation error for each sub-type
            # that fails to validate
            msg = "3 validation errors for graph"
        else:
            msg = "In function graph: Expected typing.Union\\[.*\\] for argument 'x', got \\[1.1\\] \\(list\\)"
        with self.assertRaisesRegex(TypeError, msg):
            build_graph(graph, [1.1])

        @csp.graph
        def graph(x: ts[typing.Union[int, float, str]]):
            pass

        build_graph(graph, csp.const(1))
        build_graph(graph, csp.const(1.1))
        build_graph(graph, csp.const("s"))
        if USE_PYDANTIC:
            msg = "cannot validate ts\\[typing.List\\[float\\]\\] as ts\\[typing.Union\\[.*\\]\\]"
        else:
            msg = "In function graph: Expected ts\\[typing.Union\\[.*\\]\\] for argument 'x', got ts\\[typing.List\\[float\\]\\]"
        with self.assertRaisesRegex(TypeError, msg):
            build_graph(graph, csp.const([1.1]))

    def test_realtime_timers(self):
        """was a bug"""
        rv = csp.run(
            csp.timer,
            timedelta(seconds=1),
            starttime=datetime.utcnow(),
            endtime=timedelta(seconds=3),
            realtime=True,
            queue_wait_time=timedelta(seconds=0.001),
        )[0]
        self.assertLess(len(rv), 3)

    def test_graph_arguments_propagation(self):
        @csp.graph
        def my_graph(s: str, i: int):
            csp.add_graph_output("s", csp.const(s))
            csp.add_graph_output("i", csp.const(i))

        rv = csp.run(my_graph, s="sss", i=42, starttime=datetime.utcnow(), endtime=timedelta(seconds=1))
        self.assertEqual(1, len(rv["s"]))
        self.assertEqual(rv["s"][0][1], "sss")
        self.assertEqual(1, len(rv["i"]))
        self.assertEqual(rv["i"][0][1], 42)

        rv = csp.run(my_graph, "sss", 42, starttime=datetime.utcnow(), endtime=timedelta(seconds=1))
        self.assertEqual(1, len(rv["s"]))
        self.assertEqual(rv["s"][0][1], "sss")
        self.assertEqual(1, len(rv["i"]))
        self.assertEqual(rv["i"][0][1], 42)

    def test_caching_non_cachable_object(self):
        class Dummy:
            def __hash__(self):
                hash({})

        @csp.graph
        def sub_graph(x: Dummy) -> csp.Outputs(i=ts[int]):
            return csp.const(1)

        @csp.graph
        def my_graph():
            csp.add_graph_output("i", sub_graph(Dummy()))

        csp.run(my_graph, starttime=datetime.now())

        @csp.graph(force_memoize=True)
        def sub_graph(x: Dummy) -> csp.Outputs(i=ts[int]):
            return csp.const(1)

        @csp.graph
        def my_graph():
            csp.add_graph_output("i", sub_graph(Dummy()))

        with self.assertRaisesRegex(TypeError, "unhashable type.*"):
            csp.run(my_graph, starttime=datetime.now())

    def test_realtime_timer_lag(self):
        '''bugfix exception when timers would reschedule on a lagged engine
        "timers in realtime raise exception if they get behind"'''
        delay = timedelta(seconds=0.25)
        timer_interval = timedelta(seconds=0.1)

        @csp.node
        def lag(x: ts[int]) -> ts[datetime]:
            if csp.ticked(x):
                import time

                time.sleep(delay.total_seconds())
                return datetime.utcnow()

        @csp.graph
        def graph(count: int, allow_deviation: bool) -> ts[datetime]:
            x = lag(csp.count(csp.timer(timer_interval, allow_deviation=allow_deviation)))
            stop_cond = csp.count(x) == count
            csp.stop_engine(csp.filter(stop_cond, stop_cond))
            return x

        results = csp.run(graph, 4, False, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)[0]
        self.assertEqual(len(results), 4)

        self.assertTrue(all((results[i][0] - results[i - 1][0]) == timer_interval for i in range(1, len(results))))
        # Assert lag from engine -> wallclock on last tick is greater than minimum expected amount
        self.assertGreater(results[-1][1] - results[-1][0], (delay - timer_interval) * len(results))

        results = csp.run(graph, 5, True, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)[0]
        self.assertEqual(len(results), 5)
        eps = timedelta()
        # Windows clock resolution is...
        if sys.platform == "win32":
            eps = timedelta(milliseconds=50)
        self.assertTrue(all((results[i][0] - results[i - 1][0]) + eps > delay for i in range(2, len(results))))

    def test_timer_exception(self):
        with self.assertRaisesRegex(ValueError, "csp.timer interval must be > 0"):
            _ = csp.timer(timedelta(0))

    def test_list_comprehension_bug(self):
        @csp.node
        def list_comprehension_bug_node(n_seconds: int, input: csp.ts["T"]) -> csp.ts[List["T"]]:
            with csp.start():
                csp.set_buffering_policy(input, tick_history=timedelta(seconds=30))

            if csp.ticked(input):
                return [csp.value_at(input, timedelta(seconds=-n_seconds + i), default=0) for i in range(1)]

        @csp.graph
        def list_comprehension_bug_graph():
            curve_int = csp.curve(int, [(timedelta(seconds=i), i) for i in range(30)])
            csp.add_graph_output("Bucket", list_comprehension_bug_node(10, curve_int))

        rv = csp.run(list_comprehension_bug_graph, starttime=datetime(2020, 1, 1))["Bucket"]
        self.assertEqual([v[1][0] for v in rv[10:]], list(range(20)))

    @unittest.skipIf(
        os.environ.get("ASAN_OPTIONS") is not None,
        reason="Test skipped when AddressSanitizer is enabled, RSS usage is much larger than usual",
    )
    def test_alarm_leak(self):
        """this was a leak in Scheduler.cpp"""

        @csp.node
        def generate(x: ts[object]):
            with csp.alarms():
                alarm = csp.alarm(str)

            if csp.ticked(x):
                for x in range(100):
                    csp.schedule_alarm(alarm, timedelta(seconds=1), "test")

        def graph():
            generate(csp.timer(timedelta(seconds=1)))

        proc_info = psutil.Process(os.getpid())
        start_mem = proc_info.memory_info().rss
        for _ in range(5):
            csp.run(graph, starttime=datetime(2020, 9, 24), endtime=timedelta(hours=1))
            gc.collect()
        end_mem = proc_info.memory_info().rss

        # 15MB leeway, the leak resulted in 50MB+ leak
        self.assertLess(end_mem - start_mem, 15000000)

    def test_multiple_alarms_bug(self):
        @csp.node
        def n() -> csp.ts[int]:
            with csp.alarms():
                a1 = csp.alarm(int)
                a2 = csp.alarm(int)
            with csp.start():
                csp.schedule_alarm(a1, timedelta(seconds=5), 1)
                csp.schedule_alarm(a2, timedelta(seconds=10), 2)
            if csp.ticked(a1):
                self.assertEqual(a1, 1)
                return a1
            if csp.ticked(a2):
                self.assertEqual(a2, 2)
                return a2

        @csp.graph
        def g():
            csp.add_graph_output("o", n())

        res = csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))
        self.assertEqual(res, {"o": [(datetime(2020, 1, 1, 0, 0, 5), 1), (datetime(2020, 1, 1, 0, 0, 10), 2)]})

    def test_memoize_non_comparable(self):
        class A:
            pass

        @csp.node
        def my_sink(x: ts["T"]):
            pass

        @csp.graph
        def g(o: object):
            my_sink(csp.const(1))

        csp.run(
            g,
            {A(): "f1", A(): "f2"},
            starttime=datetime(
                2020,
                1,
                1,
            ),
            endtime=timedelta(seconds=1),
        )
        csp.run(
            g,
            {A(), A()},
            starttime=datetime(
                2020,
                1,
                1,
            ),
            endtime=timedelta(seconds=1),
        )

    def test_nested_using(self):
        @csp.graph
        def g(x: "~X", y: "~Y"):
            pass

        csp.run(g.using(X=int).using(Y=float), 1, 2, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))
        with self.assertRaises(TypeError):
            csp.run(g.using(X=int).using(Y=str), 1, 2, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))

    def test_null_nodes(self):
        @csp.node
        def assert_never_ticks(i: ts["T"]):
            if csp.ticked(i):
                raise RuntimeError("Unexpected ticked value")

        @csp.graph
        def g():
            assert_never_ticks.using(T=str)(csp.null_ts(str))
            assert_never_ticks(csp.null_ts(str))
            with self.assertRaises(TypeError):
                assert_never_ticks.using(T=int)(csp.null_ts(str))

        csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=10))

    def test_start_end_times(self):
        start_time = datetime(2020, 1, 1, 9, 31, 5, 1)
        end_time = start_time + timedelta(seconds=20)

        @csp.node
        def n():
            with csp.start():
                self.assertEqual(csp.engine_start_time(), start_time)
                self.assertEqual(csp.engine_end_time(), end_time)

        @csp.graph
        def g():
            self.assertEqual(csp.engine_start_time(), start_time)
            self.assertEqual(csp.engine_end_time(), end_time)
            n()

        csp.run(g, starttime=start_time, endtime=end_time)

        with self.assertRaisesRegex(RuntimeError, "csp graph information is not available"):
            csp.engine_start_time()

    # SIGINT wont work on windows ( https://docs.python.org/3/library/os.html#os.kill ), may not be worth the trouble to make this test work on windows
    @unittest.skipIf(sys.platform == "win32", "tests needs windows port")
    def test_ctrl_c(self):
        pid = os.fork()
        if pid == 0:
            all_good = False
            try:
                x = csp.timer(timedelta(seconds=1), True)
                csp.run(x, starttime=datetime.utcnow(), endtime=timedelta(seconds=60), realtime=True)
            except KeyboardInterrupt:
                all_good = True

            os._exit(all_good)
        else:
            import signal
            import time

            time.sleep(1)
            os.kill(pid, signal.SIGINT)
            all_good = os.waitpid(pid, 0)

            self.assertTrue(all_good)

    def test_curve_multiple_values_same_time(self):
        '''addresses "Add support for multiple values on same timestamp for csp.curve"'''

        @csp.graph
        def g() -> csp.Outputs(o1=csp.ts[int], o2=csp.ts[int], o3=csp.ts[int]):
            values = [
                (timedelta(seconds=0), 0),
                (timedelta(seconds=0), 1),
                (timedelta(seconds=1), 2),
                (timedelta(seconds=2), 3),
                (timedelta(seconds=3), 4),
                (timedelta(seconds=3), 5),
                (timedelta(seconds=4), 6),
                (timedelta(seconds=5), 7),
                (timedelta(seconds=5), 8),
            ]
            return csp.output(
                o1=csp.curve(int, values),
                o2=csp.curve(int, values, push_mode=PushMode.NON_COLLAPSING),
                o3=csp.curve(int, values, push_mode=PushMode.LAST_VALUE),
            )

        res = csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=5))
        for k in ("o1", "o2"):
            times, values = zip(*res[k])
            self.assertEqual(
                times, tuple(datetime(2020, 1, 1) + timedelta(seconds=s) for s in [0, 0, 1, 2, 3, 3, 4, 5, 5])
            )
            self.assertEqual(values, tuple(range(9)))
        times, values = zip(*res["o3"])
        self.assertEqual(times, tuple(datetime(2020, 1, 1) + timedelta(seconds=s) for s in [0, 1, 2, 3, 4, 5]))
        self.assertEqual(values, (1, 2, 3, 5, 6, 8))

    def test_engine_scheduling_order(self):
        @csp.node
        def my_node(val: int) -> ts[int]:
            with csp.alarms():
                a = csp.alarm(int)
            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=0), val)
            if csp.ticked(a):
                csp.schedule_alarm(a, timedelta(seconds=1), val)
                return a

        @csp.node
        def dummy(v: ts[int]) -> ts[int]:
            return v

        @csp.graph
        def my_ranked_node(val: int, rank: int = 0) -> csp.Outputs(val=ts[int]):
            res = my_node(val)
            for i in range(rank):
                res = dummy(res)
            return csp.output(val=res)

        @csp.graph
        def my_graph():
            n1 = csp.curve(int, [(timedelta(seconds=i), 1) for i in range(6)])
            n2 = my_ranked_node(3).val
            n3 = my_ranked_node(6, 4).val
            n4 = my_ranked_node(5, 3).val
            n5 = csp.curve(int, [(timedelta(seconds=i), 2) for i in range(6)])
            n6 = my_ranked_node(4).val
            csp.add_graph_output("o", csp.collect([n1, n2, n3, n4, n5, n6]))

        def verify_res(res):
            for t, l in res:
                self.assertEqual(l, [1, 2, 3, 4, 5, 6])

        verify_res(csp.run(my_graph, starttime=datetime(2020, 1, 1), endtime=timedelta(seconds=5))["o"])

    def test_datetime_timedelta_ranges(self):
        """range check when converting datetime"""
        for d in [
            datetime(2020, 12, 24, 1, 2, 3, 123456),
            datetime(1970, 1, 1),
            # Negative Epochs times are not supported on windows
            datetime(1969, 5, 6, 2, 3, 4) if sys.platform != "win32" else datetime(1970, 1, 1),
            datetime(1969, 5, 6, 2, 3, 4, 123456) if sys.platform != "win32" else datetime(1970, 1, 1),
            # Edge cases, DateTime MIN / MAX
            datetime(1678, 1, 1) if sys.platform == "linux" else datetime(1970, 1, 1),
            datetime(2261, 12, 31, 23, 59, 59, 999999),
            timedelta(days=1, seconds=3600, microseconds=123456),
            timedelta(days=-1, seconds=3600, microseconds=123456),
        ]:
            res = csp.run(csp.const(d), starttime=datetime(2020, 12, 24))[0][0][1]
            self.assertEqual(res, d, f"date: {d}")

        # Out of bounds
        with self.assertRaisesRegex(OverflowError, "datetime 1677-09-20 00:00:00 is out of range for csp datetime"):
            d1 = datetime(1677, 9, 20)
            csp.run(csp.const(d1), starttime=datetime(2020, 12, 24))

        with self.assertRaisesRegex(OverflowError, "datetime 2262-04-12 00:00:00 is out of range for csp datetime"):
            d2 = datetime(2262, 4, 12)
            csp.run(csp.const(d2), starttime=datetime(2020, 12, 24))

        with self.assertRaisesRegex(OverflowError, "timedelta 106752 days, 0:00:00 out of range for csp timedelta"):
            td = timedelta(days=106752)
            csp.run(csp.const(td), starttime=datetime(2020, 12, 24))

        with self.assertRaisesRegex(OverflowError, "timedelta -106752 days, 0:00:00 out of range for csp timedelta"):
            td = timedelta(days=-106752)
            csp.run(csp.const(td), starttime=datetime(2020, 12, 24))

    def test_realtime_endtime(self):
        from csp.impl.pushadapter import PushInputAdapter
        from csp.impl.wiring import py_push_adapter_def

        # Ensure engine exits at end time even if no events are coming in ( ensure its not exiting due to the queue wait time setting )
        adapter = py_push_adapter_def("adapter", PushInputAdapter, ts[int])
        csp.run(
            adapter(),
            starttime=datetime.utcnow(),
            endtime=timedelta(seconds=0.5),
            realtime=True,
            queue_wait_time=timedelta(days=1),
        )

    def test_start_realtime_in_future(self):
        import pytz

        t = datetime.now(pytz.UTC) + timedelta(seconds=1)
        res = csp.run(csp.const(123), starttime=t, endtime=t, realtime=True)[0][0]
        self.assertEqual(res[1], 123)

    def test_threaded_run(self):
        # simple test
        runner = csp.run_on_thread(
            csp.count, csp.timer(timedelta(seconds=1)), starttime=datetime(2021, 4, 23), endtime=timedelta(seconds=60)
        )
        res = runner.join()[0]
        self.assertEqual(len(res), 60)
        # ensure stopping doesnt try to access dead push input adapter
        runner.stop_engine()

        # realtime
        @csp.graph
        def g(count: int) -> csp.ts[int]:
            x = csp.count(csp.timer(timedelta(seconds=0.1)))
            stop = x == count
            stop = csp.filter(stop, stop)

            csp.stop_engine(stop)
            return x

        runner = csp.run_on_thread(g, 5, starttime=datetime.utcnow(), endtime=timedelta(seconds=60), realtime=True)
        res = runner.join()[0]
        self.assertEqual(len(res), 5)

        # midway stop
        runner = csp.run_on_thread(g, 50000, starttime=datetime.utcnow(), endtime=timedelta(minutes=30), realtime=True)
        import time

        time.sleep(1)
        runner.stop_engine()
        res = runner.join()[0]
        self.assertLess(len(res), 1000)

        # exception handling
        @csp.node
        def err(x: ts[object]):
            if csp.ticked(x) and csp.num_ticks(x) > 5:
                a = b

        runner = csp.run_on_thread(err, csp.timer(timedelta(seconds=0.01)), realtime=True)
        with self.assertRaisesRegex(RuntimeError, ""):
            runner.join()

    def test_int_to_float_ts_conversion(self):
        @csp.node
        def eq(i: csp.ts["T1"], f: csp.ts["T2"]):
            self.assertEqual(float(i), float(f))

        @csp.node
        def basket_wrapper(l: [csp.ts[float]], d: {str: csp.ts[float]}) -> csp.Outputs(
            l=csp.OutputBasket(List[csp.ts[float]], shape_of="l"),
            d=csp.OutputBasket(Dict[str, csp.ts[float]], shape_of="d"),
        ):
            if csp.ticked(l):
                ticked_value_types = set(map(type, l.tickedvalues()))
                self.assertEqual(len(ticked_value_types), 1)
                self.assertIs(next(iter(ticked_value_types)), float)
                csp.output(l=dict(l.tickeditems()))
            if csp.ticked(d):
                ticked_value_types = set(map(type, d.tickedvalues()))
                self.assertEqual(len(ticked_value_types), 1)
                self.assertIs(next(iter(ticked_value_types)), float)
                csp.output(d=dict(d.tickeditems()))

        def g():
            c_int = csp.count(csp.timer(timedelta(seconds=1)))
            c_float = csp.sample.using(T=float)(c_int, c_int)
            eq(c_int, c_float)
            basket_outputs = basket_wrapper([c_int], {"0": c_int, "1": c_int})
            eq(basket_outputs.l[0], c_float)
            eq(basket_outputs.d["0"], c_float)
            eq(basket_outputs.d["1"], c_float)

        csp.run(
            g,
            starttime=datetime(
                2021,
                1,
                1,
            ),
            endtime=timedelta(seconds=10),
        )

    def test_outputs_with_dict_naming(self):
        # This was a bug where outputs named with dict properties, ie "values", would fail to be accessed on the Outputs object
        @csp.node
        def foo(x: ts[int]) -> csp.Outputs(values=ts[int]):
            csp.output(values=x)

        def g():
            rv = foo(csp.const(5))
            return rv.values

        rv = csp.run(
            g,
            starttime=datetime(
                2021,
                1,
                1,
            ),
            endtime=timedelta(seconds=10),
        )
        self.assertEqual(rv[0][0][1], 5)

    def test_pass_none_as_ts(self):
        @csp.node
        def n(a: csp.ts[int], d: {int: csp.ts[int]}, l: [csp.ts[int]]) -> csp.ts[int]:
            if csp.ticked(a):
                return a + d[0] + l[0]

        @csp.graph
        def g(a: csp.ts[int], d: {int: csp.ts[int]}, l: [csp.ts[int]]) -> csp.ts[int]:
            if a is not None:
                return a + d[0] + l[0]
            else:
                assert d is None
                assert l is None
                return csp.const.using(T=int)(-1)

        @csp.graph
        def main(use_graph: bool, pass_null: bool) -> csp.Outputs(o=csp.ts[int]):
            inst = g if use_graph else n

            if pass_null:
                return csp.output(o=inst(None, None, None))
            else:
                return csp.output(o=inst(csp.const(1), {0: csp.const(2)}, [csp.const(3)]))

        res1 = csp.run(
            main,
            True,
            False,
            starttime=datetime(
                2021,
                1,
                1,
            ),
            endtime=timedelta(seconds=10),
        )
        self.assertEqual(res1["o"][0][1], 6)
        res2 = csp.run(
            main,
            True,
            True,
            starttime=datetime(
                2021,
                1,
                1,
            ),
            endtime=timedelta(seconds=10),
        )
        self.assertEqual(res2["o"][0][1], -1)
        res3 = csp.run(
            main,
            False,
            False,
            starttime=datetime(
                2021,
                1,
                1,
            ),
            endtime=timedelta(seconds=10),
        )
        self.assertEqual(res3["o"][0][1], 6)
        with self.assertRaises(TypeError):
            csp.run(
                main,
                False,
                True,
                starttime=datetime(
                    2021,
                    1,
                    1,
                ),
                endtime=timedelta(seconds=10),
            )

    def test_return_arg_mismatch(self):
        @csp.graph
        def my_graph(x: csp.ts[int]) -> csp.ts[str]:
            return x

        with self.assertRaises(TypeError) as ctxt:
            csp.run(my_graph, csp.const(1), starttime=datetime.utcnow())
        if USE_PYDANTIC:
            self.assertIn(
                "cannot validate ts[int] as ts[str]: <class 'int'> is not a subclass of <class 'str'>",
                str(ctxt.exception),
            )
        else:
            self.assertEqual(
                str(ctxt.exception), "In function my_graph: Expected ts[str] for return value, got ts[int]"
            )

        @csp.graph
        def dictbasket_graph(x: csp.ts[int]) -> Dict[str, csp.ts[str]]:
            return csp.output({"a": x})

        if USE_PYDANTIC:
            msg = re.escape("cannot validate ts[int] as ts[str]: <class 'int'> is not a subclass of <class 'str'>")
        else:
            msg = (
                "In function dictbasket_graph: Expected typing\.Dict\[str, .* for return value, got \{'a': .* \(dict\)"
            )
        with self.assertRaisesRegex(TypeError, msg):
            csp.run(dictbasket_graph, csp.const(1), starttime=datetime.utcnow())

        @csp.graph
        def listbasket_graph(x: csp.ts[int]) -> List[csp.ts[str]]:
            return csp.output([x])

        if USE_PYDANTIC:
            msg = re.escape("cannot validate ts[int] as ts[str]: <class 'int'> is not a subclass of <class 'str'>")
        else:
            msg = "In function listbasket_graph: Expected typing\.List\[.* for return value, got \[.* \(list\)"
        with self.assertRaisesRegex(TypeError, msg):
            csp.run(listbasket_graph, csp.const(1), starttime=datetime.utcnow())

    def test_global_context(self):
        try:

            @csp.graph
            def g() -> csp.ts[int]:
                return csp.const(1)

            res1 = csp.run(
                g,
                starttime=datetime(
                    2021,
                    1,
                    1,
                ),
                endtime=timedelta(seconds=10),
            )
            c = g()
            res2 = csp.run(
                c,
                starttime=datetime(
                    2021,
                    1,
                    1,
                ),
                endtime=timedelta(seconds=10),
            )

            self.assertEqual(res1, res2)

            replace_b_with_c = False

            @csp.graph
            def g() -> csp.Outputs(a=csp.ts[int], b=csp.ts[int]):
                key = csp.curve(
                    str, [(timedelta(seconds=i), v) for i, v in enumerate(["A", "B", "A", "A", "A", "B", "C", "B"])]
                )
                value = csp.curve(int, [(timedelta(seconds=i), i) for i in range(8)])

                demux = csp.DelayedDemultiplex(value, key)
                a = demux.demultiplex("A")
                if replace_b_with_c:
                    b = demux.demultiplex("C")
                else:
                    b = demux.demultiplex("B")
                return csp.output(a=a, b=b)

            res1 = csp.run(
                g,
                starttime=datetime(
                    2021,
                    1,
                    1,
                ),
                endtime=timedelta(seconds=10),
            )
            with self.assertRaisesRegex(RuntimeError, ".*Delayed node must be created under a wiring context.*"):
                outputs = g()
            csp.new_global_context()
            outputs = g()
            res2a = csp.run(
                outputs.a,
                starttime=datetime(
                    2021,
                    1,
                    1,
                ),
                endtime=timedelta(seconds=10),
            )
            self.assertEqual(res2a[0], res1["a"])
            res2b = csp.run(
                outputs.b,
                starttime=datetime(
                    2021,
                    1,
                    1,
                ),
                endtime=timedelta(seconds=10),
            )
            self.assertEqual(res2b[0], res1["b"])
            csp.clear_global_context()
            with self.assertRaisesRegex(RuntimeError, ".*Delayed node must be created under a wiring context.*"):
                outputs = g()
            c = csp.new_global_context(False)
            with self.assertRaisesRegex(RuntimeError, ".*Delayed node must be created under a wiring context.*"):
                outputs = g()
            with c:
                outputs = g()
                res2a = csp.run(
                    outputs.a,
                    starttime=datetime(
                        2021,
                        1,
                        1,
                    ),
                    endtime=timedelta(seconds=10),
                )
                self.assertEqual(res2a[0], res1["a"])
            with self.assertRaisesRegex(RuntimeError, ".*Delayed node must be created under a wiring context.*"):
                outputs = g()
        finally:
            csp.clear_global_context()

    def test_unnamed_basket_return(self):
        @csp.node
        def n(x: {str: csp.ts["T"]}) -> csp.OutputBasket(Dict[str, csp.ts["T"]], shape_of="x"):
            if csp.ticked(x):
                return csp.output({k: v for k, v in x.tickeditems()})

        @csp.node
        def n2(x: [csp.ts["T"]]) -> csp.OutputBasket(List[csp.ts["T"]], shape_of="x"):
            if csp.ticked(x):
                return csp.output({k: v for k, v in x.tickeditems()})

        def g():
            res = n({"a": csp.const("v1"), "b": csp.const("v2")})
            res2 = n2(list(res.values()))
            csp.add_graph_output("a", res["a"])
            csp.add_graph_output("b", res["b"])
            csp.add_graph_output("c", res2[0])
            csp.add_graph_output("d", res2[1])

        res = csp.run(g, starttime=datetime(2021, 1, 1), endtime=timedelta(seconds=1))
        self.assertEqual(
            res,
            {
                "a": [(datetime(2021, 1, 1, 0, 0), "v1")],
                "b": [(datetime(2021, 1, 1, 0, 0), "v2")],
                "c": [(datetime(2021, 1, 1, 0, 0), "v1")],
                "d": [(datetime(2021, 1, 1, 0, 0), "v2")],
            },
        )

    def test_delayed_edge(self):
        x = csp.DelayedEdge(ts[int])
        with self.assertRaisesRegex(RuntimeError, "Encountered unbound DelayedEdge"):
            csp.run(x, starttime=datetime.utcnow(), endtime=timedelta())

        self.assertFalse(x.is_bound())
        x.bind(csp.const(123))
        self.assertTrue(x.is_bound())

        res = csp.run(x, starttime=datetime.utcnow(), endtime=timedelta())[0][0][1]
        self.assertEqual(res, 123)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Attempted to bind DelayedEdge multiple times, previously bound to output from node \"csp.const\"",
        ):
            x.bind(csp.const(456))

        # Type check
        if USE_PYDANTIC:
            msg = r"""cannot validate ts[int] as ts[str]: <class 'int'> is not a subclass of <class 'str'>"""
        else:
            msg = r"""Expected ts[T] for argument 'edge', got ts[int](T=str)"""
        with self.assertRaisesRegex(TypeError, re.escape(msg)):
            y = csp.DelayedEdge(ts[str])
            y.bind(csp.const(123))

        # Null default
        z = csp.DelayedEdge(ts[int], default_to_null=True)
        self.assertFalse(z.is_bound())
        res = csp.run(z, starttime=datetime.utcnow(), endtime=timedelta())[0]
        self.assertEqual(len(res), 0)
        z.bind(csp.const(123))
        res = csp.run(z, starttime=datetime.utcnow(), endtime=timedelta())[0][0][1]
        self.assertEqual(res, 123)

        # Should raise at this point
        with self.assertRaisesRegex(
            RuntimeError,
            r"Attempted to bind DelayedEdge multiple times, previously bound to output from node \"csp.const\"",
        ):
            x.bind(csp.const(456))

    def test_cyclical_graph_error(self):
        """Ensure cyclical graphs generate clear errors, can occur with delayed bindings"""

        def g():
            a = csp.DelayedCollect(int)
            b = csp.DelayedCollect(int)

            a.add_input(csp.unroll(b.output()))
            b.add_input(csp.unroll(a.output()))

            csp.add_graph_output("a", csp.unroll(a.output()) - 5)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Illegal cycle found in graph, path:\n\t\*\* unroll -> collect -> unroll -> collect -> unroll \*\*  -> _binary_op -> GraphOutputAdapter",
        ):
            csp.run(g, starttime=datetime.utcnow(), endtime=timedelta())

    def test_delayed_edge_derived_type(self):
        class Base(csp.Struct):
            a: int

        class Derived(Base):
            b: float

        test_self = self

        class MyDelayedNode(DelayedNodeWrapperDef):
            def __init__(self):
                super().__init__()
                self.output = csp.DelayedEdge(csp.ts[Base])

            def copy(self):
                raise NotImplementedError()

            def _instantiate(self):
                with test_self.assertRaises(TypeError):
                    self.output.bind(csp.const(1))

                self.output.bind(csp.const(Derived(a=1, b=2)))

        @csp.graph
        def g() -> csp.ts[Base]:
            return MyDelayedNode().output

        res = csp.run(g, starttime=datetime.utcnow(), endtime=timedelta())[0][0][1]
        self.assertEqual(res, Derived(a=1, b=2))

    def test_realtime_flag(self):
        def g(expected_realtime: bool):
            self.assertEqual(expected_realtime, csp.is_configured_realtime())

        csp.run(g, False, starttime=datetime.utcnow(), endtime=timedelta())
        csp.run(g, True, starttime=datetime.utcnow(), endtime=timedelta(), realtime=True)

    def test_graph_shape_bug(self):
        """Address an assertion error bug that we had on returning list baskets with specified shape"""

        @csp.graph
        def aux(x: [ts[float]], y: {str: ts[float]}) -> csp.Outputs(
            o1=csp.OutputBasket(List[ts[float]], shape_of="x"),
            o2=csp.OutputBasket(Dict[str, ts[float]], shape_of="y"),
        ):
            return csp.output(o1=x, o2=y)

        @csp.graph
        def g() -> csp.Outputs(o1=csp.OutputBasket(List[ts[float]]), o2=csp.OutputBasket(Dict[str, ts[float]])):
            res = aux([csp.const(1.0), csp.const(2.0)], {"3": csp.const(3.0), "4": csp.const(4.0)})
            return csp.output(o1=res.o1, o2=res.o2)

        res = csp.run(g, starttime=datetime.now(), endtime=timedelta(seconds=10))
        self.assertEqual([v[0][1] for v in res.values()], [1.0, 2.0, 3.0, 4.0])

    def test_graph_node_pickling(self):
        """Checks for a bug that we had when transitioning to python 3.8 - the graphs and nodes became unpicklable
        :return:
        """
        from csp.tests.test_engine import _dummy_graph, _dummy_node

        self.assertEqual(_dummy_graph, pickle.loads(pickle.dumps(_dummy_graph)))
        self.assertEqual(_dummy_node, pickle.loads(pickle.dumps(_dummy_node)))

    def test_memoized_object(self):
        @csp.csp_memoized
        def my_data():
            return object()

        @csp.node
        def my_node() -> csp.ts[object]:
            with csp.alarms():
                a = csp.alarm(bool)
            with csp.start():
                csp.schedule_alarm(a, timedelta(), True)
            if csp.ticked(a):
                return my_data()

        @csp.graph
        def g() -> csp.Outputs(o1=csp.ts[object], o2=csp.ts[object]):
            return csp.output(o1=csp.const(my_data()), o2=my_node())

        res = csp.run(g, starttime=datetime.now(), endtime=timedelta(seconds=10))
        self.assertEqual(id(res["o1"][0][1]), id(res["o2"][0][1]))

    def test_separate_graph_build_and_run(self):
        @csp.graph
        def g() -> csp.ts[int]:
            return csp.const(1)

        s = datetime.now()
        e = timedelta(seconds=10)

        g_built1 = build_graph(g)
        # Should be fine to run graph that was built without start or end time
        csp.run(g_built1, starttime=s, endtime=e)

        with self.assertRaisesRegex(
            AssertionError,
            "Start time and end time should either both be specified or none of them should be specified when building a graph",
        ):
            g_built1 = build_graph(g, starttime=s)
        with self.assertRaisesRegex(
            AssertionError,
            "Start time and end time should either both be specified or none of them should be specified when building a graph",
        ):
            g_built1 = build_graph(g, endtime=e)

        g_built2 = build_graph(g, starttime=s, endtime=e)
        # Both of those should be fine
        csp.run(g_built2, starttime=s, endtime=e)
        csp.run(g_built2, starttime=s, endtime=s + e)

        with self.assertRaisesRegex(AssertionError, "Trying to run graph on period.*while it was built for.*"):
            csp.run(g_built2, starttime=s, endtime=e + timedelta(seconds=1))

    def test_graph_kwargs_return(self):
        @csp.node
        def f(x: ts[int]) -> csp.Outputs(a=ts[int], b=ts[int]):
            if csp.ticked(x):
                return csp.output(a=x, b=x + 2)

        @csp.graph
        def g(x: ts[int]) -> csp.Outputs(a=ts[int], b=ts[int]):
            return csp.output(**f(x))

        res = csp.run(g, csp.const(1), starttime=datetime.utcnow(), endtime=timedelta())
        self.assertEqual(res["a"][0][1], 1)
        self.assertEqual(res["b"][0][1], 3)

        # Error testing
        with self.assertRaisesRegex(
            csp.CspParseError, "only unpacking of other csp.node or csp.graph calls are allowed"
        ):

            @csp.graph
            def g2() -> csp.Outputs(x=ts[int]):
                return csp.output(**{})

        with self.assertRaisesRegex(csp.CspParseError, "f outputs dont align with graph outputs"):

            @csp.graph
            def g3() -> csp.Outputs(x=ts[int]):
                return csp.output(**f(csp.const(1)))

        def some_func():
            return {}

        with self.assertRaisesRegex(
            csp.CspParseError, "only unpacking of other csp.node or csp.graph calls are allowed"
        ):

            @csp.graph
            def g4() -> csp.Outputs(x=ts[int]):
                return csp.output(**some_func())

        pass

    def test_scheduler_exception(self):
        '''was a bug "scheduler accounting is off when callbacks throw"'''

        from csp.impl.pulladapter import PullInputAdapter
        from csp.impl.wiring import py_pull_adapter_def

        class MyPullAdapterImpl(PullInputAdapter):
            def __init__(self):
                self._next_time = None
                self._c = 0
                super().__init__()

            def start(self, start_time, end_time):
                super().start(start_time, end_time)
                self._next_time = start_time

            def next(self):
                self._c += 1
                time = self._next_time
                self._next_time += timedelta(seconds=1)
                if self._c > 10:
                    raise RuntimeError("all good")
                return time, self._c

        MyPullAdapter = py_pull_adapter_def("MyPullAdapter", MyPullAdapterImpl, ts[int])

        @csp.node
        def my_node(x: ts[object]):
            with csp.alarms():
                a = csp.alarm(object)

            if csp.ticked(x):
                csp.schedule_alarm(a, timedelta(seconds=1), [1, 2, 3, "a"])

        def g():
            with csp.memoize(False):
                for i in range(3):
                    c = csp.count(csp.timer(timedelta(seconds=1), True))
                    my_node(c)
                    data = MyPullAdapter()
                    csp.add_graph_output(str(i), data)

        with self.assertRaisesRegex(RuntimeError, "all good"):
            csp.run(g, starttime=datetime(2023, 1, 1), endtime=timedelta(1))

    def test_stop_cannot_be_called_without_start(self):
        """
        Was a BUG where one node raising an exception during its start block led to other nodes exeucting their stop block without ever starting
        """
        status = {"foo_started": False, "foo_stopped": False, "bar_started": False, "bar_stopped": False}

        # Python nodes (use try-finally logic for stopping)

        @csp.node
        def foo():
            with csp.start():
                status["foo_started"] = True
                raise RuntimeError("foo!")

            with csp.stop():
                status["foo_stopped"] = True

        @csp.node
        def bar():
            with csp.start():
                status["bar_started"] = True

            with csp.stop():
                status["bar_stopped"] = True

        @csp.graph
        def my_graph():
            foo()
            bar()

        with self.assertRaises(RuntimeError):
            csp.run(my_graph, realtime=True)

        self.assertTrue(status["foo_started"] and not status["foo_stopped"])
        self.assertFalse(status["bar_started"] or status["bar_stopped"])

        # C++ nodes (stop is initiated by the engine)

        class RunInfo:
            def __init__(self):
                self.n1_started = False
                self.n2_started = False
                self.n1_stopped = False
                self.n2_stopped = False

        @csp.node(cppimpl=_csptestlibimpl.start_n1_set_value)
        def n1(obj_: RunInfo):
            return

        @csp.node(cppimpl=_csptestlibimpl.start_n2_throw)
        def n2(obj_: RunInfo):
            return

        myd = RunInfo()

        @csp.graph
        def g():
            n1(myd)
            n2(myd)

        with self.assertRaises(ValueError):
            csp.run(g, realtime=True)

        self.assertTrue(myd.n1_started and myd.n1_stopped)
        self.assertFalse(myd.n2_started or myd.n2_stopped)

        # Test case where node starts, never ticks, and then stops
        status = {"started": False, "stopped": False}

        @csp.node
        def n3(x: ts[int]) -> ts[int]:
            with csp.start():
                status["started"] = True
            with csp.stop():
                status["stopped"] = True

            return 0

        @csp.graph
        def g() -> ts[int]:
            return n3(csp.null_ts(int))

        csp.run(g, starttime=datetime(2020, 1, 1), endtime=timedelta())
        self.assertTrue(status["started"] and status["stopped"])

    # SIGINT wont work on windows ( https://docs.python.org/3/library/os.html#os.kill ), may not be worth the trouble to make this test work on windows
    @unittest.skipIf(sys.platform == "win32", "tests needs windows port")
    def test_interrupt_stops_all_nodes(self):
        @csp.node
        def n(l: list, idx: int):
            with csp.stop():
                l[idx] = True

        @csp.node
        def raise_interrupt():
            with csp.alarms():
                a = csp.alarm(bool)
            with csp.start():
                csp.schedule_alarm(a, timedelta(seconds=1), True)
            if csp.ticked(a):
                import signal

                os.kill(os.getpid(), signal.SIGINT)

        # Python nodes
        @csp.graph
        def g(l: list):
            n(l, 0)
            n(l, 1)
            n(l, 2)
            raise_interrupt()

        stopped = [False, False, False]
        with self.assertRaises(KeyboardInterrupt):
            csp.run(g, stopped, starttime=datetime.utcnow(), endtime=timedelta(seconds=60), realtime=True)

        for element in stopped:
            self.assertTrue(element)

        # C++ nodes
        class RTI:
            def __init__(self):
                self.stopped = [False, False, False]

        @csp.node(cppimpl=_csptestlibimpl.set_stop_index)
        def n2(obj_: object, idx: int):
            return

        @csp.graph
        def g2(rti: RTI):
            n2(rti, 0)
            n2(rti, 1)
            n2(rti, 2)
            raise_interrupt()

        rti = RTI()
        with self.assertRaises(KeyboardInterrupt):
            csp.run(g2, rti, starttime=datetime.utcnow(), endtime=timedelta(seconds=60), realtime=True)

        for element in rti.stopped:
            self.assertTrue(element)


if __name__ == "__main__":
    unittest.main()
