import random
import time
import unittest
from datetime import datetime, timedelta

import csp
from csp import basketlib, ts


class TestBasket(unittest.TestCase):
    def setUp(self):
        self.sync_threshold = timedelta(minutes=1)

    def test_cppimpl(self):
        @csp.node
        def random_gen(trigger: ts[object]) -> ts[float]:
            if csp.ticked(trigger):
                csp.output(1000 * random.random())

        @csp.graph
        def test_graph():
            trigger1 = csp.timer(timedelta(minutes=1))  # Exactly every minute
            trigger2 = csp.timer(
                timedelta(minutes=1) + timedelta(seconds=random.random() * 2 - 1)
            )  # Every minute +/- 1 second

            random_floats1 = random_gen(trigger1)
            random_floats_async = random_gen(trigger2)

            # test _basket_synchronize_list
            synced_py = basketlib.sync_list_internal.python(
                x=[random_floats1, random_floats_async],
                trigger=csp.null_ts(bool),
                threshold=self.sync_threshold,
                output_incomplete=True,
                use_trigger=False,
            )
            synced_cpp = basketlib.sync_list_internal(
                x=[random_floats1, random_floats_async],
                trigger=csp.null_ts(bool),
                threshold=self.sync_threshold,
                output_incomplete=True,
                use_trigger=False,
            )

            synced_auto_list = basketlib.sync(
                x=[random_floats1, random_floats_async], threshold=self.sync_threshold, output_incomplete=True
            )
            synced_auto_dict = basketlib.sync(
                x={"a": random_floats1, "b": random_floats_async}, threshold=self.sync_threshold, output_incomplete=True
            )
            synced_int_dict = basketlib.sync(
                x={0: random_floats1, 1: random_floats_async}, threshold=self.sync_threshold, output_incomplete=True
            )

            csp.add_graph_output("synced_py", synced_py[1])
            csp.add_graph_output("synced_cpp", synced_cpp[1])
            csp.add_graph_output("synced_auto_list", synced_auto_list[1])
            csp.add_graph_output("synced_auto_dict", synced_auto_dict["b"])
            csp.add_graph_output("synced_int_dict", synced_int_dict[1])

        seed = time.time()
        print(f"Seeding with {seed}")
        random.seed(seed)
        results = csp.run(test_graph, starttime=datetime(2022, 6, 17, 9, 30), endtime=timedelta(minutes=390))

        self.assertEqual(results["synced_py"], results["synced_cpp"])
        self.assertEqual(results["synced_py"], results["synced_auto_list"])
        self.assertEqual(results["synced_auto_list"], results["synced_auto_dict"])
        self.assertEqual(results["synced_auto_dict"], results["synced_int_dict"])

    def test_basic(self):
        @csp.graph()
        def basic_graph():
            a = csp.curve(
                typ=float,
                data=[
                    (datetime(2022, 6, 17, 9, 30), 1.0),
                    (datetime(2022, 6, 17, 9, 45), 2.0),
                    (datetime(2022, 6, 17, 9, 50), 3.0),
                ],
            )
            b = csp.curve(
                typ=float,
                data=[
                    (datetime(2022, 6, 17, 9, 35), 4.0),
                    (datetime(2022, 6, 17, 9, 46), 5.0),
                    (datetime(2022, 6, 17, 9, 50), 6.0),
                ],
            )
            c = csp.curve(
                typ=float,
                data=[
                    (datetime(2022, 6, 17, 9, 40), 7.0),
                    (datetime(2022, 6, 17, 9, 47), 8.0),
                    (datetime(2022, 6, 17, 9, 50), 9.0),
                ],
            )

            synced = basketlib.sync(x=[a, b, c], threshold=self.sync_threshold, output_incomplete=True)
            synced_complete = basketlib.sync(x=[a, b, c], threshold=self.sync_threshold, output_incomplete=False)

            csp.add_graph_output("synced_0", synced[0])
            csp.add_graph_output("synced_1", synced[1])
            csp.add_graph_output("synced_2", synced[2])
            csp.add_graph_output("synced_complete_0", synced_complete[0])
            csp.add_graph_output("synced_complete_1", synced_complete[1])
            csp.add_graph_output("synced_complete_2", synced_complete[2])

        results = csp.run(basic_graph, starttime=datetime(2022, 6, 17, 9, 30), endtime=timedelta(30))

        self.assertEqual(
            results["synced_0"],
            [
                (datetime(2022, 6, 17, 9, 31), 1.0),
                (datetime(2022, 6, 17, 9, 46), 2.0),
                (datetime(2022, 6, 17, 9, 50), 3.0),
            ],
        )

        self.assertEqual(
            results["synced_1"],
            [
                (datetime(2022, 6, 17, 9, 36), 4.0),
                (datetime(2022, 6, 17, 9, 46), 5.0),
                (datetime(2022, 6, 17, 9, 50), 6.0),
            ],
        )

        self.assertEqual(
            results["synced_2"],
            [
                (datetime(2022, 6, 17, 9, 41), 7.0),
                (datetime(2022, 6, 17, 9, 48), 8.0),
                (datetime(2022, 6, 17, 9, 50), 9.0),
            ],
        )

        self.assertEqual(results["synced_complete_0"], [(datetime(2022, 6, 17, 9, 50), 3.0)])
        self.assertEqual(results["synced_complete_1"], [(datetime(2022, 6, 17, 9, 50), 6.0)])
        self.assertEqual(results["synced_complete_2"], [(datetime(2022, 6, 17, 9, 50), 9.0)])

    def test_sync_basket_with_trigger(self):
        st = datetime(2020, 1, 1)

        @csp.graph
        def graph():
            trigger = csp.curve(
                typ=str,
                data=[
                    (st + timedelta(seconds=10), "trigger-1"),  # regular case
                    (st + timedelta(seconds=30), "trigger-2"),  # exact same tick as a, includes it
                    (st + timedelta(seconds=31), "trigger-ignored"),  # in an active period, ignored
                    (st + timedelta(seconds=50), "trigger-3"),  # only a will get a tick here
                ],
            )
            a = csp.curve(
                typ=float,
                data=[
                    (st + timedelta(seconds=1), 1.0),
                    (st + timedelta(seconds=12), 2.0),
                    (st + timedelta(seconds=30), 3.0),
                    (st + timedelta(seconds=52), 4.0),
                ],
            )
            b = csp.curve(
                typ=float,
                data=[
                    (st + timedelta(seconds=2), 5.0),
                    (st + timedelta(seconds=14), 6.0),
                    (st + timedelta(seconds=32), 7.0),
                    (st + timedelta(seconds=56), 8.0),
                ],
            )

            synced_list = basketlib.sync_list(x=[a, b], threshold=timedelta(seconds=5), trigger=trigger)
            synced_dict = basketlib.sync_dict(x={"a": a, "b": b}, threshold=timedelta(seconds=5), trigger=trigger)

            csp.add_graph_output("list_a", synced_list[0])
            csp.add_graph_output("list_b", synced_list[1])
            csp.add_graph_output("dict_a", synced_dict["a"])
            csp.add_graph_output("dict_b", synced_dict["b"])

        result = csp.run(graph, starttime=st, endtime=st + timedelta(minutes=1))

        expected_a = [
            (st + timedelta(seconds=14), 2.0),
            (st + timedelta(seconds=32), 3.0),
            (st + timedelta(seconds=55), 4.0),
        ]
        expected_b = [(st + timedelta(seconds=14), 6.0), (st + timedelta(seconds=32), 7.0)]
        print(result["list_a"])
        self.assertEqual(result["list_a"], expected_a)
        self.assertEqual(result["list_b"], expected_b)
        self.assertEqual(result["dict_a"], expected_a)
        self.assertEqual(result["dict_b"], expected_b)

    def test_sample_dict_basket(self):
        @csp.graph
        def graph():
            trigger = csp.timer(timedelta(seconds=7), True)
            a = csp.curve(typ=float, data=[(datetime(2020, 1, 1), 1.0), (datetime(2020, 1, 1, 0, 0, 8), 2.0)])
            b = csp.curve(
                typ=float,
                data=[
                    (datetime(2020, 1, 1), 3.0),
                    (datetime(2020, 1, 1, 0, 0, 49), 4.0),
                    (datetime(2020, 1, 1, 0, 0, 50), 6.0),
                ],
            )
            c = csp.curve(typ=float, data=[(datetime(2022, 6, 17, 9, 40), 7.0)])
            sampled_x = basketlib.sample_dict(trigger=trigger, x={"a": a, "b": b, "c": c})
            csp.add_graph_output("sampled_dict_b", sampled_x["b"])

        st = datetime(2020, 1, 1)
        result = csp.run(graph, starttime=st, endtime=st + timedelta(minutes=1))
        ans = (
            [(st + timedelta(seconds=i * 7), 3.0) for i in range(1, 7)]
            + [(st + timedelta(seconds=49), 4.0)]
            + [(st + timedelta(seconds=56), 6.0)]
        )
        self.assertEqual(result["sampled_dict_b"], ans)

    def test_sample_list_basket(self):
        @csp.graph
        def graph():
            trigger = csp.timer(timedelta(seconds=7), True)
            a = csp.curve(typ=float, data=[(datetime(2020, 1, 1), 1.0), (datetime(2020, 1, 1, 0, 0, 8), 2.0)])
            b = csp.curve(
                typ=float,
                data=[
                    (datetime(2020, 1, 1), 3.0),
                    (datetime(2020, 1, 1, 0, 0, 49), 4.0),
                    (datetime(2020, 1, 1, 0, 0, 50), 6.0),
                ],
            )
            c = csp.curve(typ=float, data=[(datetime(2022, 6, 17, 9, 40), 7.0)])
            sampled_x = basketlib.sample_basket(trigger=trigger, x=[a, b, c])
            csp.add_graph_output("sampled_list_a", sampled_x[0])
            csp.add_graph_output("sampled_list_c", sampled_x[2])

        st = datetime(2020, 1, 1)
        result = csp.run(graph, starttime=st, endtime=st + timedelta(minutes=1))
        ans = [(st + timedelta(seconds=7), 1.0)] + [(st + timedelta(seconds=i * 7), 2.0) for i in range(2, 9)]
        self.assertEqual(result["sampled_list_a"], ans)
        self.assertEqual(result["sampled_list_c"], [])

    def test_cppimpl_basket_sample(self):
        @csp.node
        def random_gen(trigger: ts[object]) -> ts[float]:
            if csp.ticked(trigger):
                csp.output(1000 * random.random())

        @csp.graph
        def test_graph():
            trigger1 = csp.timer(timedelta(minutes=1))  # Exactly every minute
            trigger2 = csp.timer(
                timedelta(minutes=1) + timedelta(seconds=random.random() * 2 - 1)
            )  # Every minute +/- 1 second

            random_floats1 = random_gen(trigger1)
            random_floats_async = random_gen(trigger2)

            sample_trigger = csp.timer(timedelta(seconds=50), True)
            synced_py = basketlib.sample_list.python(trigger=sample_trigger, x=[random_floats1, random_floats_async])
            synced_cpp = basketlib.sample_list(trigger=sample_trigger, x=[random_floats1, random_floats_async])

            csp.add_graph_output("synced_periodic_py", synced_py[0])
            csp.add_graph_output("synced_periodic_cpp", synced_cpp[0])
            csp.add_graph_output("synced_aperiodic_py", synced_py[1])
            csp.add_graph_output("synced_aperiodic_cpp", synced_cpp[1])

        seed = 9740
        random.seed(seed)
        results = csp.run(test_graph, starttime=datetime(2022, 6, 17, 9, 30), endtime=timedelta(minutes=390))

        self.assertEqual(results["synced_periodic_py"], results["synced_periodic_cpp"])
        self.assertEqual(results["synced_aperiodic_py"], results["synced_aperiodic_cpp"])


if __name__ == "__main__":
    unittest.main()
