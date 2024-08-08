import threading
import time
import unittest
from datetime import datetime, timedelta

import csp
from csp import PushMode, ts
from csp.impl.pushpulladapter import PushGroup, PushPullInputAdapter
from csp.impl.wiring import py_pushpull_adapter_def


class MyPushPullAdapter(PushPullInputAdapter):
    def __init__(self, typ, data):
        self._data = data
        self._thread = None
        self._exc = None
        self._running = False

    def start(self, starttime, endtime):
        self._running = True
        self._thread = threading.Thread(target=self.run)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()
        if self._exc:
            raise self._exc

    def run(self):
        try:
            self._run()
        except Exception as exc:
            self._exc = exc

    def _run(self):
        idx = 0
        sent_live = False
        while self._running and idx < len(self._data):
            live, t, v = self._data[idx]
            idx += 1
            diff = t - datetime.utcnow()
            if diff > timedelta(seconds=0):
                time.sleep(diff.total_seconds())
            self.push_tick(live, t, v)
            if live:
                sent_live = True
        if not sent_live:
            self.flag_replay_complete()


test_adapter = py_pushpull_adapter_def("test", MyPushPullAdapter, ts["T"], typ="T", data=list)


class TestPushPullAdapter(unittest.TestCase):
    def test_basic(self):
        class Data(csp.Struct):
            time: datetime
            value: int
            live: bool

        @csp.node
        def check(x: ts[object]):
            print(csp.now(), x)

        def graph(num_sim, num_rt, num_sim_repeats):
            start_time = datetime.utcnow() - (num_sim + 1) * timedelta(minutes=1)
            data = []
            for x in range(num_sim):
                for c in range(num_sim_repeats):
                    t = start_time + timedelta(minutes=x)
                    data.append((False, t, Data(time=t, value=x, live=False)))

            rtstart = datetime.utcnow() + timedelta(seconds=1)
            for x in range(num_sim, num_sim + num_rt):
                t = rtstart + timedelta(seconds=0.1 * x)
                data.append((True, t, Data(time=t, value=x, live=True)))

            nc = test_adapter(Data, data, push_mode=PushMode.NON_COLLAPSING)
            lv = test_adapter(Data, data, push_mode=PushMode.LAST_VALUE)
            # csp.print( 'nc', ts )
            csp.add_graph_output("nc", nc)
            csp.add_graph_output("lv", lv)

            expected_ticks = num_sim * 2 + num_rt

            stop = csp.count(nc) == expected_ticks
            csp.stop_engine(csp.filter(stop, stop))

        num_sim = 10
        num_rt = 10
        num_sim_repeats = 2

        res = csp.run(
            graph,
            num_sim,
            num_rt,
            num_sim_repeats,
            starttime=datetime.utcnow() - timedelta(hours=1),
            endtime=datetime.utcnow() + timedelta(seconds=30),
            realtime=True,
        )
        nc = res["nc"]
        lv = res["lv"]

        nc_sim = list(v for v in nc if not v[1].live)
        nc_live = list(v for v in nc if v[1].live)

        lv_sim = list(v for v in lv if not v[1].live)

        self.assertEqual(len(nc_sim), num_sim * num_sim_repeats)
        self.assertEqual(len(nc_live), num_rt)

        self.assertEqual(len(lv_sim), num_sim)

        for t, v in nc_sim:
            self.assertEqual(t, v.time)

        for t, v in nc_live:
            self.assertGreater(t, v.time)

    def test_out_of_order(self):
        """test ticking sim after live"""

        def graph():
            dt = datetime.utcnow()
            data = [(False, dt, 1), (True, dt, 2), (False, dt, 3)]

            return test_adapter(int, data)

        with self.assertRaisesRegex(RuntimeError, "tried to push a sim tick after live tick"):
            csp.run(graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10))

    def test_historical(self):
        """test only sim data"""

        def graph():
            dt = datetime.utcnow()
            data = [(False, dt, 1), (False, dt, 2), (False, dt, 3)]
            return test_adapter(int, data)

        graph_out = csp.run(graph, starttime=datetime.utcnow(), endtime=timedelta(seconds=10), realtime=False)
        result = [out[1] for out in graph_out[0]]
        self.assertEqual(result, [1, 2, 3])

    def test_adapter_engine_shutdown(self):
        class MyPushPullAdapterImpl(PushPullInputAdapter):
            def __init__(self, typ, data, shutdown_before_live):
                self._data = data
                self._thread = None
                self._running = False
                self._shutdown_before_live = shutdown_before_live

            def start(self, starttime, endtime):
                self._running = True
                self._thread = threading.Thread(target=self._run)
                self._thread.start()

            def stop(self):
                if self._running:
                    self._running = False
                    self._thread.join()

            def _run(self):
                idx = 0
                while self._running and idx < len(self._data):
                    if idx and self._shutdown_before_live:
                        time.sleep(0.1)
                        self.shutdown_engine(ValueError("Dummy exception message"))
                    t, v = self._data[idx]
                    self.push_tick(False, t, v)
                    idx += 1
                self.flag_replay_complete()

                idx = 0
                while self._running:
                    self.push_tick(True, datetime.utcnow(), len(self._data) + 1)
                    if idx and not self._shutdown_before_live:
                        time.sleep(0.1)
                        self.shutdown_engine(TypeError("Dummy exception message"))
                    idx += 1

        MyPushPullAdapter = py_pushpull_adapter_def(
            "MyPushPullAdapter", MyPushPullAdapterImpl, ts["T"], typ="T", data=list, shutdown_before_live=bool
        )

        @csp.graph
        def graph(shutdown_before_live: bool):
            data = [(datetime(2020, 1, 1, 2), 1), (datetime(2020, 1, 1, 3), 2)]
            adapter = MyPushPullAdapter(int, data, shutdown_before_live)
            csp.print("adapter", adapter)

        with self.assertRaisesRegex(ValueError, "Dummy exception message"):
            csp.run(graph, True, starttime=datetime(2020, 1, 1, 1))
        with self.assertRaisesRegex(TypeError, "Dummy exception message"):
            csp.run(
                graph,
                False,
                starttime=datetime(2020, 1, 1, 1),
                endtime=datetime.utcnow() + timedelta(seconds=2),
                realtime=True,
            )


if __name__ == "__main__":
    unittest.main()
