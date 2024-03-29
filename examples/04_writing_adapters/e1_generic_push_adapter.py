"""
csp.GenericPushAdapter is a more user-friendly and the simplest way to generically push some stream of data
from a non-csp engine thread into the csp engine.
"""

import threading
import time
from datetime import datetime, timedelta

import csp


# This will be run be some separate thread
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
        print("driver thread started")
        counter = 0
        # Optionally, we can wait for the adapter to start before proceeding
        # Alternatively we can start pushing data, but push_tick may fail and return False if
        # the csp engine isn't ready yet
        self._adapter.wait_for_start()

        while self._active and not self._adapter.stopped():
            self._adapter.push_tick(counter)
            counter += 1
            time.sleep(1)


@csp.graph
def my_graph():
    adapter = csp.GenericPushAdapter(int)
    driver = Driver(adapter)
    driver.start()

    # Lets be nice and shutdown the driver thread when the engine is done
    csp.schedule_on_engine_stop(driver.stop)
    csp.print("data", adapter.out())


def main():
    csp.run(my_graph, realtime=True, starttime=datetime.utcnow(), endtime=timedelta(seconds=2))


if __name__ == "__main__":
    main()
