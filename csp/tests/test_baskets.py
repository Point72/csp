import random
import time
import unittest
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List

import numpy

import csp
import csp.impl
from csp import ts


class TestBaskets(unittest.TestCase):
    def test_functionality(self):
        @csp.node
        def list_basket(x: List[ts[int]]) -> csp.Outputs(
            tickedvalues=ts[List[int]],
            tickeditems=ts[List[object]],
            tickedkeys=ts[List[int]],
            validvalues=ts[List[int]],
            validitems=ts[List[object]],
            validkeys=ts[List[int]],
            valid=ts[bool],
            iter=ts[List[int]],
            elem_access=ts[int],
        ):
            if csp.ticked(x):
                csp.output(valid=csp.valid(x))
                csp.output(tickedvalues=list(x.tickedvalues()))
                csp.output(tickedkeys=list(x.tickedkeys()))
                csp.output(tickeditems=list(x.tickeditems()))
                csp.output(validvalues=list(x.validvalues()))
                csp.output(validkeys=list(x.validkeys()))
                csp.output(validitems=list(x.validitems()))

                if csp.valid(x):
                    csp.output(iter=list(x))

                if csp.ticked(x[1]):
                    csp.output(elem_access=x[1])

        @csp.node
        def dict_basket(x: Dict[str, ts[int]]) -> csp.Outputs(
            tickedvalues=ts[List[int]],
            tickeditems=ts[List[object]],
            tickedkeys=ts[List[str]],
            validvalues=ts[List[int]],
            validitems=ts[List[object]],
            validkeys=ts[List[str]],
            valid=ts[bool],
            elem_access=ts[int],
        ):
            if csp.ticked(x):
                self.assertEqual(x.keys(), list("ABCD"))

                csp.output(valid=csp.valid(x))
                csp.output(tickedvalues=list(x.tickedvalues()))
                csp.output(tickedkeys=list(x.tickedkeys()))
                csp.output(tickeditems=list(x.tickeditems()))
                csp.output(validvalues=list(x.validvalues()))
                csp.output(validkeys=list(x.validkeys()))
                csp.output(validitems=list(x.validitems()))

                if csp.ticked(x["B"]):
                    csp.output(elem_access=x["B"])

        x0 = csp.curve(int, [(timedelta(seconds=v), v) for v in range(1, 10)])
        x1 = csp.curve(int, [(timedelta(seconds=v * 0.5), v) for v in range(1, 20)])
        x2 = csp.curve(int, [(timedelta(seconds=v * 0.25), v) for v in range(1, 40)])
        x3 = csp.curve(int, [(timedelta(seconds=v * 0.125), v) for v in range(1, 80)])

        # We expect x0 to tick every second, x1 every .5, x2 every .25 and x3 every .125
        st = datetime(2020, 2, 7, 9)

        for node, args in zip([list_basket, dict_basket], [[x0, x1, x2, x3], {"A": x0, "B": x1, "C": x2, "D": x3}]):
            result = csp.run(node, args, starttime=st)
            input_keys = list(args.keys() if isinstance(args, dict) else range(len(args)))

            valid = []
            validkeys = []
            validvalues = []
            validitems = []
            tickedvalues = []
            tickedkeys = []
            tickeditems = []

            interval = timedelta(seconds=0.125)
            t = st + interval
            # Easiest way to tet is to compute our expectations
            for x in range(1, 80):
                valid.append((t, x >= 8))
                validk = []
                validv = []

                if x >= 8:
                    validk.append(input_keys[0])
                    validv.append(x // 8)
                if x >= 4:
                    validk.append(input_keys[1])
                    validv.append(x // 4)
                if x >= 2:
                    validk.append(input_keys[2])
                    validv.append(x // 2)

                validk.append(input_keys[3])
                validv.append(x)
                validkeys.append((t, validk))
                validvalues.append((t, validv))
                validitems.append((t, list(zip(validk, validv))))

                tickedk = []
                tickedv = []
                if x % 8 == 0:
                    tickedk.append(input_keys[0])
                    tickedv.append(x // 8)
                if x % 4 == 0:
                    tickedk.append(input_keys[1])
                    tickedv.append(x // 4)
                if x % 2 == 0:
                    tickedk.append(input_keys[2])
                    tickedv.append(x // 2)
                tickedk.append(input_keys[3])
                tickedv.append(x)

                tickedkeys.append((t, tickedk))
                tickedvalues.append((t, tickedv))
                tickeditems.append((t, list(zip(tickedk, tickedv))))
                t += interval

            self.assertEqual(result["valid"], valid)
            self.assertEqual(result["validkeys"], validkeys)
            self.assertEqual(result["validvalues"], validvalues)
            self.assertEqual(result["validitems"], validitems)
            self.assertEqual(result["tickedkeys"], tickedkeys)
            self.assertEqual(result["tickedvalues"], tickedvalues)
            self.assertEqual(result["tickeditems"], tickeditems)

            self.assertEqual([x[1] for x in result["elem_access"]], list(range(1, 20)))
            if "iter" in result:
                self.assertEqual(result["iter"], validvalues[7:])

    def test_input_after_basket(self):
        @csp.node
        def last_intraday_price_calc(basket: [csp.ts[int]], enabled: csp.ts[bool]):
            if csp.ticked(enabled) and csp.valid(enabled) and enabled:
                pass
            else:
                pass

        def build_graph():
            last_intraday_price_calc([csp.const(1), csp.const(2)], csp.const(True))

        # This was a bug.
        # UnboundLocalError: local variable 'enabled' referenced before assignment
        csp.run(build_graph, starttime=datetime.now())

    def test_dynamic_baskets(self):
        @csp.node
        def gen(x: ts[object], t: "T") -> csp.Outputs(
            dyn=csp.DynamicBasket[str, "T"], changes=ts[list], ticks=ts[list]
        ):
            # Note we intentionally pass 'T' to invoke tvar code as part of the test
            with csp.state():
                s_keys = set()

            if csp.ticked(x):
                ticks = []
                changes = []
                if len(s_keys) and random.random() < 0.3:
                    count = random.randint(0, len(s_keys) // 2)
                    for _ in range(count):
                        key = numpy.random.choice(list(sorted(s_keys)))
                        s_keys.remove(key)

                        # Test both modes
                        if random.random() < 0.5:
                            csp.output(dyn={key: csp.impl.REMOVE_DYNAMIC_KEY})
                        else:
                            csp.remove_dynamic_key(dyn, key)

                        changes.append({"added": False, "key": key})

                count = random.randint(1, 10)
                keys = numpy.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), count, replace=False)

                for key in keys:
                    if key not in s_keys:
                        changes.append({"added": True, "key": key})
                    s_keys.add(key)

                    data = (key, csp.num_ticks(x))
                    csp.output(dyn={data[0]: data[1]})
                    ticks.append(data)

                # This test a logical flaw that existed in the original impl where a key ticks, a different key is removed
                # and the ticked key gets reshuffled ( but the tick index wasnt updated properly )
                # Delete a key that wasnt added this round
                ticked = set(e[0] for e in ticks)
                oldkeys = s_keys - (s_keys & ticked)
                if len(oldkeys) and random.random() < 0.2:
                    key = oldkeys.pop()
                    s_keys.remove(key)
                    csp.remove_dynamic_key(dyn, key)
                    changes.append({"added": False, "key": key})

                csp.output(changes=changes, ticks=ticks)

        @csp.node
        def consume(x: Dict[ts[str], ts[float]], changes: ts[list], ticks: ts[list]):
            with csp.state():
                s_valid = {}

            # test shape ticked
            if len(changes):
                self.assertTrue(csp.ticked(x.shape))
                self.assertEqual(len(changes), len(x.shape.events))
            else:
                self.assertFalse(csp.ticked(x.shape))

            # test shape changes
            for event, devent in zip(changes, x.shape.events):
                self.assertEqual(event["added"], devent.added)
                self.assertEqual(event["key"], devent.key)
                if not event["added"]:
                    s_valid.pop(event["key"])
                else:
                    s_valid[event["key"]] = None

            # check ticked items iteration
            tickeditems = list(x.tickeditems())
            self.assertEqual(len(ticks), len(tickeditems))
            for tick, item in zip(ticks, tickeditems):
                self.assertEqual(tick, item)
                s_valid[tick[0]] = tick[1]

                # assert invidial csp.tiucked checks
                self.assertTrue(csp.ticked(x[tick[0]]))

            # check valid items iteration
            validitems = list(x.validitems())
            self.assertEqual(len(s_valid), len(validitems))
            for k in s_valid.keys():
                self.assertEqual(s_valid[k], x[k])

            # check keys
            self.assertEqual(sorted(x.keys()), list(sorted(s_valid.keys())))

        def g():
            d = gen(csp.timer(timedelta(seconds=1)), float)
            consume(d.dyn, d.changes, d.ticks)

        seed = int(time.time())
        print(f"Using seed {seed}")
        numpy.random.seed(seed)
        random.seed(seed)
        csp.run(g, starttime=datetime(2021, 5, 25), endtime=timedelta(hours=4))

    def test_dynamic_basket_tick_remove_exception(self) -> csp.OutputBasket(Dict[ts[str], ts[int]]):
        # tick/remove exception check
        @csp.node
        def gen() -> csp.OutputBasket(Dict[ts[str], ts[int]]):
            with csp.alarms():
                a_same_cycle_check = csp.alarm(bool)

            with csp.start():
                csp.schedule_alarm(a_same_cycle_check, timedelta(), True)

            if csp.ticked(a_same_cycle_check):
                csp.output({"FOOBAR": 1})
                csp.remove_dynamic_key("FOOBAR")

        @csp.node
        def consume(x: Dict[ts[str], ts[float]]):
            pass

        with self.assertRaisesRegex(
            RuntimeError, "Attempted to delete dynamic basket key 'FOOBAR' which was already ticked this cycle"
        ):
            csp.run(consume, gen(), starttime=datetime(2021, 5, 25), endtime=timedelta(seconds=1))

    def test_dynamic_basket_buffering_policy(self):
        @csp.node
        def gen(x: ts[object]) -> csp.OutputBasket(Dict[ts[str], ts[int]]):
            with csp.state():
                s_last = defaultdict(lambda: 1)

            if csp.ticked(x):
                key = random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
                v = s_last[key]
                s_last[key] += 1
                csp.output({key: v})

        @csp.node
        def consume(x: Dict[ts[str], ts[int]]):
            with csp.start():
                csp.set_buffering_policy(x, tick_count=10, tick_history=timedelta(seconds=30))

            if csp.ticked(x):
                for k in x.tickedkeys():
                    self.assertEqual(csp.value_at(x[k], 0), csp.num_ticks(x[k]))
                    v = csp.value_at(x[k], -2, default=None)
                    if v is not None:
                        self.assertEqual(v, csp.num_ticks(x[k]) - 2)
                    _ = csp.value_at(x[k], -timedelta(seconds=10), default=None)

        csp.run(
            consume, gen(csp.timer(timedelta(seconds=1))), starttime=datetime(2021, 5, 25), endtime=timedelta(hours=1)
        )

    def test_basket_valid(self):
        # a basket input that is passive from the start should still register as valid once all its ts have ticked
        @csp.node
        def triggered(x: Dict[str, ts[float]], y: ts[bool]) -> ts[float]:
            with csp.start():
                csp.make_passive(x)

            if csp.ticked(y) and csp.valid(x):
                return sum(b for _a, b in x.validitems())

        start_time = datetime(2020, 1, 3)
        end_time = start_time + timedelta(seconds=4)

        def g():
            p = csp.curve(float, [(start_time + timedelta(seconds=i), i) for i in range(5)])
            q = csp.curve(float, [(start_time + timedelta(seconds=i), 10 * i) for i in range(5)])
            t = csp.curve(bool, [(start_time + timedelta(seconds=3), True)])
            res = triggered({"p": p, "q": q}, t)
            return res

        result = csp.run(g, starttime=start_time, endtime=end_time)
        self.assertEqual(result[0], [(start_time + timedelta(seconds=3), 33)])

    def test_list_basket_np_index(self):
        @csp.node
        def echo_np(x: List[ts[float]], num_keys: int) -> csp.OutputBasket(List[ts[float]], shape="num_keys"):
            if csp.ticked(x):
                all_idx = numpy.arange(num_keys)
                return dict(zip(all_idx, x))

        @csp.node
        def echo_int(x: List[ts[float]], num_keys: int) -> csp.OutputBasket(List[ts[float]], shape="num_keys"):
            if csp.ticked(x):
                all_idx = numpy.arange(num_keys)
                return dict(zip(all_idx.tolist(), x))  # Converts idx from np.int64 -> int

        @csp.graph
        def list_basket_graph():
            a = csp.const(1)
            b = csp.const(2)
            c = csp.const(3)
            list_basket = [a, b, c]

            out_int = echo_int(list_basket, 3)
            out_np = echo_np(list_basket, 3)

            csp.add_graph_output("int_index_0", out_int[0])
            csp.add_graph_output("int_index_1", out_int[1])
            csp.add_graph_output("int_index_2", out_int[2])

            csp.add_graph_output("np_index_0", out_np[0])
            csp.add_graph_output("np_index_1", out_np[1])
            csp.add_graph_output("np_index_2", out_np[2])

        result = csp.run(list_basket_graph, starttime=datetime.now(), endtime=timedelta(seconds=10))
        self.assertEqual(result["int_index_0"], result["np_index_0"])
        self.assertEqual(result["int_index_1"], result["np_index_1"])
        self.assertEqual(result["int_index_2"], result["np_index_2"])


if __name__ == "__main__":
    unittest.main()
