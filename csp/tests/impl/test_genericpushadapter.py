import threading
import time
import unittest
from datetime import datetime, timedelta

import csp


class Driver:
    def __init__(self, adapter: csp.GenericPushAdapter):
        self._adapter = adapter
        self._active = False
        self._thread = None

    def start(self):
        self._active = True
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def stop(self):
        if self._active:
            self._active = False
            self._thread.join()

    def _run(self):
        counter = 0
        self._adapter.wait_for_start()

        while self._active and not self._adapter.stopped():
            self._adapter.push_tick(counter)
            counter += 1
            time.sleep(0.001)


class TestGenericPushAdapter(unittest.TestCase):
    def test_basic(self):
        def graph():
            adapter = csp.GenericPushAdapter(int, name="Generic")
            driver = Driver(adapter)
            driver.start()
            csp.schedule_on_engine_stop(driver.stop)

            x = adapter.out()
            stop = csp.count(x) == 50
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return x

        res = csp.run(
            graph,
            starttime=datetime.utcnow(),
            endtime=timedelta(1),
            realtime=True,
            queue_wait_time=timedelta(seconds=0),
        )[0]
        self.assertEqual(len(res), 50)
        self.assertEqual([x[1] for x in res], list(range(50)))


if __name__ == "__main__":
    unittest.main()
