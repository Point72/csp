import itertools
import random
import string
import time
import unittest
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import numpy

import csp
from csp import ts


class DynData(csp.Struct):
    key: str
    val: int


@csp.node
def gen_basket(keys: ts[List[str]], deletes: ts[List[str]]) -> csp.DynamicBasket[str, DynData]:
    with csp.state():
        s_counts = defaultdict(int)
    if csp.ticked(keys):
        for key in keys:
            s_counts[key] += 1
            csp.output({key: DynData(key=key, val=s_counts[key])})

    if csp.ticked(deletes):
        for key in deletes:
            csp.remove_dynamic_key(key)


@csp.node
def random_keys(keys: List[str], interval: timedelta, repeat: bool) -> ts[List[str]]:
    with csp.alarms():
        x = csp.alarm(int)
    with csp.state():
        s_keys = list(keys)

    with csp.start():
        csp.schedule_alarm(x, interval, 0)

    if csp.ticked(x):
        count = min(random.randint(1, 5), len(s_keys))
        res = list(numpy.random.choice(s_keys, count, replace=False))
        if not repeat:
            for k in res:
                s_keys.remove(k)

        if s_keys:
            csp.schedule_alarm(x, interval, 0)

        return res


@csp.node
def delayed_deletes(keys: ts[List[str]], delay: timedelta) -> ts[List[str]]:
    with csp.alarms():
        delete = csp.alarm(List[str])
    with csp.state():
        s_pending = set()

    if csp.ticked(keys):
        deletes = []
        for k in keys:
            if k not in s_pending:
                s_pending.add(k)
                deletes.append(k)
        if deletes:
            csp.schedule_alarm(delete, delay, deletes)

    if csp.ticked(delete):
        for key in delete:
            s_pending.remove(key)
        return delete


class TestDynamic(unittest.TestCase):
    def setUp(self):
        seed = int(time.time())
        print("SEEDING with", seed)
        numpy.random.seed(seed)
        random.seed(seed)

    def test_start_stop_dynamic(self):
        started = {}
        stopped = {}

        @csp.node
        def start_stop(key: str, x: ts[object]):
            with csp.start():
                started[key] = True

            with csp.stop():
                stopped[key] = True

            if csp.ticked(x):
                pass

        @csp.graph
        def dyn_graph(key: str):
            start_stop(key, csp.const(1))

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), True)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn_graph, csp.snapkey())
            csp.add_graph_output("keys", keys)

        res = csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))
        actual_keys = set(itertools.chain.from_iterable(v[1] for v in res["keys"]))
        self.assertEqual(set(started.keys()), actual_keys)
        self.assertEqual(set(stopped.keys()), actual_keys)

    def test_clean_shutdown(self):
        """ensure inputs are cleaned up on shutdown by triggering events past shutdown"""

        @csp.graph
        def dyn_graph(key: str):
            # const delay
            csp.add_graph_output(f"{key}_const", csp.merge(csp.const(1), csp.const(2, delay=timedelta(seconds=10))))

            # timer
            csp.add_graph_output(f"{key}_timer", csp.timer(timedelta(seconds=1)))

            # pull adapter
            data = [(timedelta(seconds=n + 1), n) for n in range(100)]
            csp.add_graph_output(f"{key}_pull", csp.curve(int, data))

            # node with alarm
            csp.add_graph_output(
                f"{key}_alarm",
                csp.merge(csp.delay(csp.const(1), timedelta(seconds=1)), csp.delay(csp.const(2), timedelta(seconds=1))),
            )

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            deletes = csp.delay(keys, timedelta(seconds=5.1))

            csp.add_graph_output("keys", keys)
            basket = gen_basket(keys, deletes)
            csp.dynamic(basket, dyn_graph, csp.snapkey())

        res = csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))
        actual_keys = set(itertools.chain.from_iterable(v[1] for v in res["keys"]))
        for key in actual_keys:
            self.assertEqual(len(res[f"{key}_const"]), 1)
            self.assertEqual(len(res[f"{key}_alarm"]), 1)
            self.assertEqual(len(res[f"{key}_timer"]), 5)
            self.assertEqual(len(res[f"{key}_pull"]), 5)

    def test_dynamic_args(self):
        """test various "special" arguments"""

        @csp.graph
        def dyn_graph(key: str, val: List[str], key_ts: ts[DynData], scalar: str):
            csp.add_graph_output(f"{key}_key", csp.const(key))
            csp.add_graph_output(f"{key}_val", csp.const(val))
            csp.add_graph_output(f"{key}_ts", key_ts)
            csp.add_graph_output(f"{key}_scalar", csp.const(scalar))

            # Lets add some actual nodes to the graph!
            # Force an initial tick so it aligns with add_graph_output data
            key_ts = csp.merge(key_ts, csp.sample(csp.const(1), key_ts))
            csp.add_graph_output(f"{key}_tsadj", key_ts.val * 2)

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), True)
            csp.add_graph_output("keys", keys)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn_graph, csp.snapkey(), csp.snap(keys), csp.attach(), "hello world!")

        res = csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))
        actual_keys = set(itertools.chain.from_iterable(v[1] for v in res["keys"]))
        for key in actual_keys:
            self.assertEqual(res[f"{key}_key"][0][1], key)
            # val is snap of list of keys when created, just assert key is in the list
            self.assertIn(key, res[f"{key}_val"][0][1])
            self.assertEqual(res[f"{key}_scalar"][0][1], "hello world!")
            ts_ticks = len(res[f"{key}_ts"])
            self.assertGreater(ts_ticks, 0)
            self.assertEqual(
                [v[1] for v in res[f"{key}_ts"]], [DynData(key=key, val=n) for n in range(1, ts_ticks + 1)]
            )
            self.assertEqual(len(res[f"{key}_tsadj"]), ts_ticks)
            self.assertTrue(all(x[1].val * 2 == y[1] for x, y in zip(res[f"{key}_ts"], res[f"{key}_tsadj"])))

    def test_shared_input(self):
        """ensure an externally wired input is shared / not recreated per sub-graph"""
        instances = []

        @csp.node
        def source_node() -> ts[int]:
            with csp.alarms():
                x = csp.alarm(int)
            with csp.start():
                print("There can be only one!")
                instances.append(1)
                csp.schedule_alarm(x, timedelta(seconds=1), 1)

            if csp.ticked(x):
                csp.schedule_alarm(x, timedelta(seconds=1), x + 1)
                return x

        @csp.graph
        def dyn_graph(key: str, x: ts[int]):
            csp.add_graph_output(key, x * x)

        def g():
            s = source_node()
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), True)
            csp.add_graph_output("keys", keys)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn_graph, csp.snapkey(), s)

        csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))
        self.assertEqual(len(instances), 1)

        ## There was a bug where nodes creates outside of run would register in the global memcache
        # and then get picked up in dynamic, which would then find it in memo.  Ensure it doesnt break
        # when passing potentially globally memoized edges directly into dynamic
        # Force memoizing csp.timer(Timedelta(seconds=1)) in global cache
        _ = csp.timer(timedelta(seconds=1))

        @csp.graph
        def dyn_graph(key: str, x: ts[int]):
            csp.add_graph_output(key, x * x)
            csp.add_graph_output(key + "_timer", csp.count(csp.timer(timedelta(seconds=1))))

        def g():
            s = csp.count(csp.timer(timedelta(seconds=1)))
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            deletes = csp.delay(keys, timedelta(seconds=2.1))
            csp.add_graph_output("keys", keys)
            basket = gen_basket(keys, deletes)
            csp.dynamic(basket, dyn_graph, csp.snapkey(), s)

        csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))

    def test_dynamic_outputs(self):
        @csp.graph
        def dyn_graph_named(key: str) -> csp.Outputs(k=ts[DynData], v=ts[int]):
            v = csp.count(csp.timer(timedelta(seconds=1)))
            k = DynData.fromts(key=csp.const(key, delay=timedelta(seconds=1)), val=v * 2)
            return csp.output(k=k, v=v)

        @csp.graph
        def dyn_graph_unnamed(key: str) -> ts[DynData]:
            v = csp.count(csp.timer(timedelta(seconds=1)))
            return DynData.fromts(key=csp.const(key, delay=timedelta(seconds=1)), val=v * 3)

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            deletes = csp.delay(keys, timedelta(seconds=5.1))
            csp.add_graph_output("keys", keys)
            basket = gen_basket(keys, deletes)
            res = csp.dynamic(basket, dyn_graph_named, csp.snapkey())
            res2 = csp.dynamic(basket, dyn_graph_unnamed, csp.snapkey())

            csp.add_graph_output("k", csp.dynamic_collect(res.k))
            csp.add_graph_output("v", csp.dynamic_collect(res.v))
            csp.add_graph_output("unnamed", csp.dynamic_collect(res2))

        res = csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))
        self.assertEqual(len(res["k"]), len(res["v"]))
        self.assertEqual(len(res["k"]), len(res["unnamed"]))

        cache = defaultdict(lambda: 1)
        for r in res["k"]:
            for k, v in r[1].items():
                self.assertEqual(k, v.key)
                self.assertEqual(cache[k] * 2, v.val)
                cache[k] += 1

        cache = defaultdict(lambda: 1)
        for r in res["v"]:
            for k, v in r[1].items():
                self.assertEqual(cache[k], v)
                cache[k] += 1

        cache = defaultdict(lambda: 1)
        for r in res["unnamed"]:
            for k, v in r[1].items():
                self.assertEqual(k, v.key)
                self.assertEqual(cache[k] * 3, v.val)
                cache[k] += 1

    def test_add_remove_add(self):
        @csp.graph
        def dyn_graph(key: str, version: int, x: ts[int]):
            key = f"{key}_{version}"
            csp.add_graph_output(key, DynData.fromts(key=csp.const(key), val=x * x))

        def g():
            c = csp.count(csp.timer(timedelta(seconds=1)))
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), True)
            # keys = csp.const.using(T=[str])( ['A'], delay=timedelta(seconds=1) )
            deletes = delayed_deletes(keys, timedelta(seconds=3.1))
            basket = gen_basket(keys, deletes)
            csp.dynamic(basket, dyn_graph, csp.snapkey(), csp.snap(csp.count(keys)), c)

        res = csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))
        for k, v in res.items():
            for tick in v:
                self.assertEqual(tick[1].key, k)

    def test_nested_dynamics(self):
        @csp.graph
        def dyn_graph_inner(parent_key: str, key: str, x: ts[int]) -> ts[DynData]:
            key = "_".join([parent_key, key])

            const_key = csp.sample(csp.firstN(x, 1), csp.const(key))
            v = DynData.fromts(key=const_key, val=x * x)
            csp.add_graph_output(key, v)
            csp.add_graph_output(key + "_alarm", csp.count(csp.timer(timedelta(seconds=1))))
            return v

        @csp.graph
        def dyn_graph(key: str, x: ts[int]) -> ts[Dict[str, DynData]]:
            keys = random_keys(list("ABC"), timedelta(seconds=0.5), False)
            deletes = delayed_deletes(keys, timedelta(seconds=1.1))
            basket = gen_basket(keys, deletes)
            res = csp.dynamic(basket, dyn_graph_inner, key, csp.snapkey(), x)
            v = csp.dynamic_collect(res)
            return v

        def g():
            c = csp.count(csp.timer(timedelta(seconds=1)))
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            deletes = delayed_deletes(keys, timedelta(seconds=3.1))

            csp.add_graph_output("keys", keys)
            basket = gen_basket(keys, deletes)
            res = csp.dynamic(basket, dyn_graph, csp.snapkey(), c)
            csp.add_graph_output("all", csp.dynamic_collect(res))

            # csp.print('keys', keys)
            # csp.print('deletes',deletes)

        res = csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=60))
        for t, v in res["all"]:
            for parent_key, child in v.items():
                for key, data in child.items():
                    self.assertEqual("_".join([parent_key, key]), data.key)

    def test_stop_engine_dynamic(self):
        @csp.node
        def on_stop(key: str, x: ts[object]):
            if csp.ticked(x):
                csp.stop_engine(dynamic=True)

        @csp.graph
        def dyn_graph(key: str) -> ts[int]:
            csp.stop_engine(csp.const(1, delay=timedelta(seconds=2)), True)
            return csp.const(1)

        @csp.node
        def assert_destruction_time(dyn_output: {ts[str]: ts[object]}):
            with csp.state():
                s_added = {}

            if csp.ticked(dyn_output.shape):
                for k in dyn_output.shape.added:
                    s_added[k] = csp.now()
                for k in dyn_output.shape.removed:
                    # Assert we were shutdown after 3 2 seconds ( csp.stop_engine call ) not after 5.1 ( key removal )
                    self.assertEqual(csp.now() - s_added[k], timedelta(seconds=2))

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            deletes = csp.delay(keys, timedelta(seconds=5.1))
            basket = gen_basket(keys, deletes)
            res = csp.dynamic(basket, dyn_graph, csp.snapkey())
            assert_destruction_time(res)

        csp.run(g, starttime=datetime(2021, 6, 22), endtime=timedelta(seconds=30))

    def test_initial_ticks(self):
        @csp.node
        def do_assert(tag: str, x: ts[object], expect_tick: bool):
            with csp.alarms():
                start = csp.alarm(int)
            with csp.state():
                s_ticked = False

            with csp.start():
                csp.schedule_alarm(start, timedelta(), True)

            if csp.ticked(start):
                self.assertEqual(csp.ticked(x), expect_tick, tag)

        @csp.node
        def assert_multiple_alarm():
            with csp.alarms():
                a = csp.alarm(bool)
            with csp.state():
                s_alarm_count = 0

            with csp.start():
                for _ in range(10):
                    csp.schedule_alarm(a, timedelta(), True)

            with csp.stop():
                self.assertEqual(s_alarm_count, 10)

            if csp.ticked(a):
                s_alarm_count += 1

        @csp.graph
        def dyn_graph(x: ts[object], y: ts[object], z: ts[object]):
            do_assert("x", x, True)  # csp.attach edge
            do_assert("y", y, True)  # csp.timer edge
            do_assert("z", z, False)  # timer not on start

            do_assert("const", csp.const(1), True)
            do_assert("curve", csp.curve(int, [(timedelta(), 1)]), True)

            assert_multiple_alarm()

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(
                basket, dyn_graph, csp.attach(), csp.timer(timedelta(seconds=1)), csp.timer(timedelta(seconds=0.17))
            )
            csp.add_graph_output("keys", keys)

        csp.run(g, starttime=datetime(2021, 6, 28), endtime=timedelta(seconds=10))

        # This was a bug where initial tick processing was dropping externally scheduled events on now that were deferred
        @csp.graph
        def dyn(s: str):
            csp.add_graph_output(f"tick_{s}", csp.const(s))

        @csp.graph
        def main():
            sym_ts = csp.flatten([csp.const("A"), csp.const("B")])
            demuxed_data = csp.dynamic_demultiplex(sym_ts, sym_ts)
            csp.dynamic(demuxed_data, dyn, csp.snapkey())
            csp.add_graph_output("sym_ts", sym_ts)

        res = csp.run(main, starttime=datetime(2021, 6, 7), endtime=timedelta(seconds=20))["sym_ts"]
        self.assertEqual([v[1] for v in res], ["A", "B"])

    def test_exceptions(self):
        # snap / attach in container
        @csp.graph
        def dyn_graph(k: List[str]):
            pass

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn_graph, [csp.snapkey()])

        with self.assertRaisesRegex(TypeError, "csp.snap and csp.attach are not supported as members of containers"):
            csp.run(g, starttime=datetime(2021, 6, 28), endtime=timedelta(seconds=10))

        # dynamic basket outputs
        @csp.graph
        def dyn_graph(k: str) -> List[ts[int]]:
            return [csp.const(1)]

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn_graph, csp.snapkey())

        with self.assertRaisesRegex(TypeError, "csp.dynamic does not support basket outputs of sub_graph"):
            csp.run(g, starttime=datetime(2021, 6, 28), endtime=timedelta(seconds=10))

        # duplicate output keys
        @csp.graph
        def dyn_graph(k: str):
            csp.add_graph_output("duplicate", csp.const(1))

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn_graph, csp.snapkey())

        with self.assertRaisesRegex(ValueError, 'graph output key "duplicate" is already bound'):
            csp.run(g, starttime=datetime(2021, 6, 28), endtime=timedelta(seconds=10))

        # csp.snap on invalid input
        @csp.graph
        def dyn_graph(snap: int):
            pass

        def g():
            keys = random_keys(list(string.ascii_uppercase), timedelta(seconds=1), False)
            basket = gen_basket(keys, csp.null_ts(List[str]))
            csp.dynamic(basket, dyn_graph, csp.snap(csp.null_ts(int)))

        with self.assertRaisesRegex(RuntimeError, "csp.snap input \\( sub_graph arg 0 \\) is not valid"):
            csp.run(g, starttime=datetime(2021, 6, 28), endtime=timedelta(seconds=10))

    def test_dynamic_with_self_reference(self):
        # This test ensures that triggers and processes can be driven by functions attached
        # to an object instance, because dynamic's do special argument parsing to inject
        # snapkey, snap, attach, etc
        class Container:
            @csp.node
            def trigger(self, x: ts[str]) -> csp.DynamicBasket[str, str]:
                if csp.ticked(x):
                    return {x: x}

            @csp.graph
            def process(self, key: str) -> ts[bool]:
                return csp.const(True)

            @csp.graph
            def main_graph(self):
                data = csp.curve(
                    str,
                    [
                        (timedelta(seconds=0), "a"),
                        (timedelta(seconds=1), "b"),
                        (timedelta(seconds=2), "a"),
                        (timedelta(seconds=3), "c"),
                        (timedelta(seconds=4), "a"),
                    ],
                )
                dyn_data = csp.dynamic(self.trigger(data), self.process, csp.snapkey())
                csp.print("Results: ", data)

        c = Container()
        csp.run(c.main_graph, starttime=datetime.utcnow().replace(microsecond=0), endtime=timedelta(seconds=10))


if __name__ == "__main__":
    unittest.main()
