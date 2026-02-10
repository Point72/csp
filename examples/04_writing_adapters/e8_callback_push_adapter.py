"""
Callback push adapter example. Unlike a regular PushInputAdapter where the adapter thread pushes
final values directly, a callback push adapter pushes callables (callbacks). The C++ engine invokes
each callable on the engine thread to produce the actual tick value. This is useful when the value
to be ticked depends on state that must be read on the engine thread, or when the production of the
value should be deferred until consumption time.

This example uses CustomPyPushInputAdapter from _csptestlibimpl, which overrides
parseCallbackEvent / deleteCallbackEvent / restoreCallbackEvent in C++ to call the
pushed Python callable and convert the result into a csp tick.
"""

import threading
import time
from datetime import timedelta

import csp
from csp import ts
from csp.impl.wiring.adapters import InputAdapterDef, _adapterdef
from csp.lib import _csptestlibimpl
from csp.utils.datetime import utc_now


class CustomPushInputAdapter(_csptestlibimpl.CustomPyPushInputAdapter):
    def start(self, starttime, endtime):
        pass

    def stop(self):
        pass


def custom_py_push_adapter_def(
    name, adapterimpl, out_type, manager_type=None, memoize=True, force_memoize=False, **kwargs
):
    def impl(mgr, engine, pytype, push_mode, scalars):
        push_group = scalars[-1]
        scalars = scalars[:-1]
        if mgr is not None:
            scalars = (mgr,) + scalars
        return _csptestlibimpl._custompushadapter(mgr, engine, pytype, push_mode, (adapterimpl, push_group, scalars))

    return _adapterdef(
        InputAdapterDef,
        name,
        impl,
        out_type,
        manager_type,
        memoize=memoize,
        force_memoize=force_memoize,
        **kwargs,
        push_group=(object, None),
    )


# The Impl object is created at runtime when the graph is converted into the runtime engine
# it does not exist at graph building time!
class MyCallbackPushAdapterImpl(CustomPushInputAdapter):
    def __init__(self, interval):
        self._interval = interval
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
        counter = 0
        while self._running:
            val = counter
            # Push a callable instead of a value — the C++ engine will invoke it
            self.push_tick(lambda val=val: {"counter": val})
            counter += 1
            time.sleep(self._interval.total_seconds())


# MyCallbackPushAdapter is the graph-building time construct.
MyCallbackPushAdapter = custom_py_push_adapter_def(
    "MyCallbackPushAdapter", MyCallbackPushAdapterImpl, ts[dict], interval=timedelta
)


@csp.graph
def my_graph():
    data = MyCallbackPushAdapter(timedelta(seconds=1))
    csp.print("data", data)


def main():
    csp.run(my_graph, realtime=True, starttime=utc_now(), endtime=timedelta(seconds=5))


if __name__ == "__main__":
    main()
