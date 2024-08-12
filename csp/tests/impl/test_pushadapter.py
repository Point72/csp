import inspect
import threading
import time
import unittest
from datetime import datetime, timedelta
from typing import List

import csp
from csp import PushMode, ts
from csp.impl.adaptermanager import AdapterManagerImpl
from csp.impl.pushadapter import PushBatch, PushGroup, PushInputAdapter
from csp.impl.wiring import py_push_adapter_def


class MyPushAdapterManager:
    def __init__(self):
        self._group = PushGroup()

    def subscribe(self, typ: type, interval: int, ticks_per_interval: int, push_mode: PushMode):
        return test_adapter(self, typ, interval, ticks_per_interval, push_mode=push_mode, push_group=self._group)

    def _create(self, engine, memo):
        return MyPushAdapterManagerImpl(engine, self)


class MyPushAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, rep):
        super().__init__(engine)
        self._rep = rep
        self._thread = None
        self._active = False
        self._engine = engine
        self._adapters = set()

    def process_next_sim_timeslice(self, now):
        return None

    def start(self, starttime, endtime):
        self._active = True
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def stop(self):
        self._active = False
        if self._thread:
            self._thread.join()

    def run(self):
        c = 0
        while self._active:
            with PushBatch(self._engine) as batch:
                for adapter in self._adapters:
                    if c % adapter._interval == 0:
                        for i in range(adapter._ticks_per_interval):
                            adapter.push_tick(c + i, batch)
            c += 1
            time.sleep(0.001)


class MyPushAdapter(PushInputAdapter):
    def __init__(self, mgrImpl, typ, interval, ticks_per_interval):
        self._type = typ
        self._interval = interval
        self._ticks_per_interval = ticks_per_interval

        mgrImpl._adapters.add(self)


test_adapter = py_push_adapter_def(
    "test", MyPushAdapter, ts["T"], MyPushAdapterManager, typ="T", interval=int, ticks_per_interval=int
)


class CurvesPushAdapterManager:
    def __init__(self, events):
        self._group = PushGroup()
        self._events = events

    def subscribe(self, typ: type, id: str, push_mode: PushMode):
        return curve_push_adapter(self, typ, id, push_mode=push_mode, push_group=self._group)

    def _create(self, engine, memo):
        return CurvesPushAdapterManagerImpl(engine, self)


class CurvesPushAdapterManagerImpl(AdapterManagerImpl):
    def __init__(self, engine, rep):
        super().__init__(engine)
        self._rep = rep
        self._engine = engine
        self._adapters = {}

    def process_next_sim_timeslice(self, now):
        return None

    def start(self, starttime, endtime):
        # We just inject everything into the queue for the test at this point
        # avoids all races
        for events in self._rep._events:
            with PushBatch(self._engine) as batch:
                for id, value in events:
                    self._adapters[id].push_tick(value, batch)


class CurvePushAdapter(PushInputAdapter):
    def __init__(self, mgrImpl, typ, id):
        mgrImpl._adapters[id] = self


curve_push_adapter = py_push_adapter_def("test", CurvePushAdapter, ts["T"], CurvesPushAdapterManager, typ="T", id=str)


class TestPushAdapter(unittest.TestCase):
    def test_basic(self):
        @csp.node
        def check(burst: ts[List["T"]], lv: List[ts["T"]], nc: ts["T"]):
            # Assert all last values have the same value since theyre injected in the same batch
            if csp.ticked(lv) and csp.valid(lv):
                self.assertTrue(all(v == lv[0] for v in lv.validvalues()))

            # Assert last values dont advance past the non-collapsing tick
            if csp.ticked(nc):
                self.assertTrue(all(v == nc for v in lv.validvalues()))
                if csp.num_ticks(nc) == 100:
                    csp.stop_engine()

            if csp.ticked(burst):
                self.assertTrue(all(v == nc for v in burst))

        def graph():
            mgr = MyPushAdapterManager()
            nc = mgr.subscribe(int, 5, 1, push_mode=PushMode.NON_COLLAPSING)
            lvb = []
            for x in range(100):
                lv = mgr.subscribe(int, 1, 1, push_mode=PushMode.LAST_VALUE)
                lvb.append(lv)

            burst = mgr.subscribe(int, 5, 1, push_mode=PushMode.BURST)
            check(burst, lvb, nc)

        csp.run(
            graph,
            starttime=datetime.utcnow(),
            endtime=timedelta(1),
            realtime=True,
            queue_wait_time=timedelta(seconds=0),
        )
        csp.run(
            graph,
            starttime=datetime.utcnow(),
            endtime=timedelta(1),
            realtime=True,
            queue_wait_time=timedelta(seconds=0.1),
        )

    def test_multiple_ticks_in_batch(self):
        # Each list in the events list is a push group of events
        events = [
            # should collapse
            [("c1", 1), ("c2", 1), ("b1", 1)],
            # collapsed inputs should sync to non-collapse
            # expects engine cycle c1:2, c2:2, nc1:1, b1:[2,3], nc2:1
            [("c1", 2), ("c2", 2), ("nc1", 1), ("nc2", 1), ("b1", 2), ("b1", 3)],
            # collapsed inputs sync, multiple ticks on nc1 unroll
            # expects cycles:
            # c1:4, c2:3, nc1:2, nc2:2, b1:[4,5,6]
            # nc1:3
            # nc1:4
            [
                ("c1", 3),
                ("c2", 3),
                ("nc1", 2),
                ("nc2", 2),
                ("nc1", 3),
                ("nc1", 4),
                ("c1", 4),
                ("b1", 4),
                ("b1", 5),
                ("b1", 6),
            ],
            # ensure this evals only after previous group is done unrolling
            # expects engine tick c1:5,c2:4,nc1:5,nc2:3,b1:[7,8,9]
            [("c1", 5), ("c2", 4), ("nc1", 5), ("nc2", 3), ("b1", 7), ("b1", 8), ("b1", 9)],
            # multiple ticks on multiple non-collapsing inputs in the same group
            # expects engine cycles
            # c1:6, c2:6, nc1:6, nc2:4, b1:[10,11]
            # nc1:7, nc2:5
            # nc2:6
            # nc2:7
            [
                ("c1", 6),
                ("c2", 5),
                ("c2", 6),
                ("b1", 10),
                ("b1", 11),
                ("nc1", 6),
                ("nc2", 4),
                ("nc1", 7),
                ("nc2", 5),
                ("nc2", 6),
                ("nc2", 7),
            ],
            # cleanup round
            [("c1", 7), ("c2", 7), ("nc1", 8), ("nc2", 8), ("b1", 12)],
        ]

        expected = [
            {"c1": 2, "c2": 2, "nc1": 1, "nc2": 1, "b1": [1, 2, 3]},
            {"c1": 4, "c2": 3, "nc1": 2, "nc2": 2, "b1": [4, 5, 6]},
            {"nc1": 3},
            {"nc1": 4},
            {"c1": 5, "c2": 4, "nc1": 5, "nc2": 3, "b1": [7, 8, 9]},
            {"c1": 6, "c2": 6, "nc1": 6, "nc2": 4, "b1": [10, 11]},
            {"nc1": 7, "nc2": 5},
            {"nc2": 6},
            {"nc2": 7},
            {"c1": 7, "c2": 7, "nc1": 8, "nc2": 8, "b1": [12]},
        ]

        @csp.node
        def b_to_d(x: {str: ts[object]}) -> ts[dict]:
            if csp.ticked(x):
                return dict(x.tickeditems())

        def graph():
            mgr = CurvesPushAdapterManager(events)
            nc1 = mgr.subscribe(int, "nc1", push_mode=PushMode.NON_COLLAPSING)
            nc2 = mgr.subscribe(int, "nc2", push_mode=PushMode.NON_COLLAPSING)

            c1 = mgr.subscribe(int, "c1", push_mode=PushMode.LAST_VALUE)
            c2 = mgr.subscribe(int, "c2", push_mode=PushMode.LAST_VALUE)

            b1 = mgr.subscribe(int, "b1", push_mode=PushMode.BURST)
            b = {"c1": c1, "c2": c2, "nc1": nc1, "nc2": nc2, "b1": b1}
            csp.add_graph_output("v", b_to_d(b))

            stop = nc2 == csp.const(8)
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)

        result = csp.run(graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=30), realtime=True)["v"]
        result = list(x[1] for x in result)
        self.assertEqual(result, expected)

    def test_adapter_engine_shutdown(self):
        class MyPushAdapterImpl(PushInputAdapter):
            def __init__(self):
                self._thread = None
                self._running = False

            def start(self, starttime, endtime):
                self._running = True
                self._thread = threading.Thread(target=self._run)
                self._thread.start()

            def stop(self):
                if self._running:
                    self._running = False
                    self._thread.join()

            def _run(self):
                pushed = False
                while self._running:
                    if pushed:
                        time.sleep(0.1)
                        self.shutdown_engine(TypeError("Dummy exception message"))
                    else:
                        self.push_tick(0)
                        pushed = True

        MyPushAdapter = py_push_adapter_def("MyPushAdapter", MyPushAdapterImpl, ts[int])

        status = {"count": 0}

        @csp.node
        def node(x: ts[object]):
            if csp.ticked(x):
                status["count"] += 1

        @csp.graph
        def graph():
            adapter = MyPushAdapter()
            node(adapter)
            csp.print("adapter", adapter)

        with self.assertRaisesRegex(TypeError, "Dummy exception message"):
            csp.run(graph, starttime=datetime.utcnow(), realtime=True)
        self.assertEqual(status["count"], 1)

    def test_help(self):
        # for `help` to work on adapters, signature must be defined
        sig = inspect.signature(test_adapter)
        self.assertEqual(sig.parameters["typ"].annotation, "T")
        self.assertEqual(sig.parameters["interval"].annotation, int)
        self.assertEqual(sig.parameters["ticks_per_interval"].annotation, int)
        self.assertEqual(sig.parameters["push_mode"].annotation, PushMode)
        self.assertEqual(sig.parameters["push_group"].annotation, object)


if __name__ == "__main__":
    unittest.main()
