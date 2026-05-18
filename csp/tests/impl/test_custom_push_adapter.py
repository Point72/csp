import threading
import time
import unittest
from datetime import timedelta

import csp
from csp import ts
from csp.impl.pushadapter import PushGroup
from csp.impl.wiring.adapters import InputAdapterDef, _adapterdef
from csp.lib import _csptestlibimpl
from csp.lib._csptestlibimpl import CustomPushBatch as PushBatch
from csp.utils.datetime import utc_now


def callable_py_push_adapter_def(
    name, adapterimpl, out_type, manager_type=None, memoize=True, force_memoize=False, **kwargs
):
    def impl(mgr, engine, pytype, push_mode, scalars):
        push_group = scalars[-1]
        scalars = scalars[:-1]
        if mgr is not None:
            scalars = (mgr,) + scalars
        return _csptestlibimpl._callablepushadapter(mgr, engine, pytype, push_mode, (adapterimpl, push_group, scalars))

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


class _CallablePushAdapterImpl(_csptestlibimpl.CallablePyPushInputAdapter):
    """Adapter impl that pushes callables from a shared list."""

    def __init__(self, typ, callables, sleep_interval=0.01):
        self._callables = callables
        self._sleep_interval = sleep_interval
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
        for c in self._callables:
            self.push_tick(c)
            if self._sleep_interval > 0:
                time.sleep(self._sleep_interval)


_CallablePushAdapter = callable_py_push_adapter_def(
    "_CallablePushAdapter", _CallablePushAdapterImpl, ts["T"], typ="T", callables=list, sleep_interval=(float, 0.01)
)


def _run_adapter(graph, timeout=5):
    """Helper to run a graph in realtime mode with standard settings."""
    return csp.run(
        graph,
        starttime=utc_now(),
        endtime=timedelta(seconds=timeout),
        realtime=True,
        queue_wait_time=timedelta(seconds=0),
    )


class _GroupCoordinator:
    """Coordinates two adapters in the same PushGroup, pushing events via PushBatch
    from a single thread to ensure deterministic ordering."""

    def __init__(self, nc_callables, lv_callables):
        self.nc_callables = nc_callables
        self.lv_callables = lv_callables
        self.nc_adapter = None
        self.lv_adapter = None
        self._engine = None
        self._thread = None
        self._ready = threading.Event()

    def register(self, adapter, role):
        if role == "nc":
            self.nc_adapter = adapter
        else:
            self.lv_adapter = adapter
        if self._engine is None:
            self._engine = adapter.engine()
        if self.nc_adapter and self.lv_adapter:
            self._ready.set()

    def start(self):
        self._thread = threading.Thread(target=self._run)
        self._thread.start()

    def _run(self):
        self._ready.wait()
        for nc_c, lv_c in zip(self.nc_callables, self.lv_callables):
            with PushBatch(self._engine) as batch:
                self.nc_adapter.push_tick(nc_c, batch)
                self.lv_adapter.push_tick(lv_c, batch)
            time.sleep(0.01)

    def stop(self):
        if self._thread:
            self._thread.join()


class _CoordinatedPushAdapterImpl(_csptestlibimpl.CallablePyPushInputAdapter):
    """Adapter impl that registers with a coordinator instead of pushing events itself."""

    def __init__(self, typ, coordinator, role):
        self._coordinator = coordinator
        self._role = role

    def start(self, starttime, endtime):
        self._coordinator.register(self, self._role)
        if self._role == "nc":
            self._coordinator.start()

    def stop(self):
        if self._role == "nc":
            self._coordinator.stop()


_CoordinatedPushAdapter = callable_py_push_adapter_def(
    "_CoordinatedPushAdapter",
    _CoordinatedPushAdapterImpl,
    ts["T"],
    typ="T",
    coordinator=object,
    role=str,
)


class TestCallablePushAdapter(unittest.TestCase):
    # --- dict tests ---

    def test_basic_callable_dict(self):
        """Push one callable returning a dict, verify output."""
        callables = [lambda: {"key": "hello"}]

        def graph():
            data = _CallablePushAdapter(dict, callables)
            stop = csp.count(data) == 1
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][1], {"key": "hello"})

    def test_callable_dict_nested(self):
        """Push a callable returning a nested dict."""
        callables = [lambda: {"a": {"b": [1, 2, 3]}, "c": True}]

        def graph():
            data = _CallablePushAdapter(dict, callables)
            stop = csp.count(data) == 1
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][1], {"a": {"b": [1, 2, 3]}, "c": True})

    def test_multiple_ticks_dict(self):
        """Push N callables returning dicts, verify all arrive in order."""
        N = 5
        callables = [lambda val=i: {"val": val} for i in range(N)]

        def graph():
            data = _CallablePushAdapter(dict, callables)
            stop = csp.count(data) == N
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), N)
        values = [r[1]["val"] for r in res]
        self.assertEqual(values, list(range(N)))

    def test_stateful_callable_dict(self):
        """Push closures sharing mutable state, verify incrementing values."""
        N = 3
        state = {"counter": 0}

        def make_callable(state=state):
            def fn(state=state):
                state["counter"] += 1
                return {"counter": state["counter"]}

            return fn

        callables = [make_callable() for _ in range(N)]

        def graph():
            data = _CallablePushAdapter(dict, callables)
            stop = csp.count(data) == N
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), N)
        values = [r[1]["counter"] for r in res]
        self.assertEqual(values, [1, 2, 3])

    # --- list tests ---

    def test_basic_callable_list(self):
        """Push one callable returning a list, verify output."""
        callables = [lambda: [1, 2, 3]]

        def graph():
            data = _CallablePushAdapter(list, callables)
            stop = csp.count(data) == 1
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][1], [1, 2, 3])

    def test_multiple_ticks_list(self):
        """Push N callables returning lists, verify all arrive in order."""
        N = 4
        callables = [lambda val=i: [val, val * 2] for i in range(N)]

        def graph():
            data = _CallablePushAdapter(list, callables)
            stop = csp.count(data) == N
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), N)
        values = [r[1] for r in res]
        self.assertEqual(values, [[0, 0], [1, 2], [2, 4], [3, 6]])

    # --- object tests ---

    def test_basic_callable_object(self):
        """Push one callable returning a tuple (via object type), verify output."""
        callables = [lambda: (1, "two", 3.0)]

        def graph():
            data = _CallablePushAdapter(object, callables)
            stop = csp.count(data) == 1
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][1], (1, "two", 3.0))

    def test_callable_object_custom_class(self):
        """Push a callable returning a custom class instance."""

        class Payload:
            def __init__(self, x, y):
                self.x = x
                self.y = y

        callables = [lambda: Payload(10, 20)]

        def graph():
            data = _CallablePushAdapter(object, callables)
            stop = csp.count(data) == 1
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), 1)
        self.assertIsInstance(res[0][1], Payload)
        self.assertEqual(res[0][1].x, 10)
        self.assertEqual(res[0][1].y, 20)

    def test_multiple_ticks_object(self):
        """Push N callables returning mixed Python objects."""
        callables = [
            lambda: 42,
            lambda: "hello",
            lambda: [1, 2],
            lambda: {"k": "v"},
        ]
        N = len(callables)

        def graph():
            data = _CallablePushAdapter(object, callables)
            stop = csp.count(data) == N
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), N)
        values = [r[1] for r in res]
        self.assertEqual(values, [42, "hello", [1, 2], {"k": "v"}])

    # --- error test ---

    def test_callable_that_raises(self):
        """Push a callable that raises, verify engine surfaces error."""

        def bad_callable():
            raise RuntimeError("test error")

        callables = [bad_callable]

        def graph():
            data = _CallablePushAdapter(dict, callables)
            csp.stop_engine(data)
            return data

        with self.assertRaises(RuntimeError):
            _run_adapter(graph)

    # --- reconsume tests ---

    def test_reconsume_non_collapsing(self):
        """Push N callables with NON_COLLAPSING mode and no sleep. Ungrouped NC adapters process
        one event per engine cycle, so remaining events are deferred to pending and reconsumed with
        reconsuming=true. Verify all arrive in order and each callable is invoked exactly once."""
        N = 5
        invocation_count = {"n": 0}
        lock = threading.Lock()

        def make_callable(val):
            def fn(val=val):
                with lock:
                    invocation_count["n"] += 1
                return {"val": val}

            return fn

        callables = [make_callable(i) for i in range(N)]

        def graph():
            data = _CallablePushAdapter(dict, callables, sleep_interval=0.0, push_mode=csp.PushMode.NON_COLLAPSING)
            stop = csp.count(data) == N
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            return data

        res = _run_adapter(graph)[0]
        self.assertEqual(len(res), N)
        values = [r[1]["val"] for r in res]
        self.assertEqual(values, list(range(N)))
        # Each callable should be invoked exactly once (not double-transformed on reconsume)
        self.assertEqual(invocation_count["n"], N)

    def test_reconsume_push_group(self):
        """Two adapters in the same PushGroup (one NC, one LAST_VALUE) pushed via PushBatch.
        The NC event locks the group at the batch boundary, deferring the LV event to pending.
        The LV event is reconsumed on the next cycle with reconsuming=true.
        Verifies isGroupEnd preservation and that each callable is invoked exactly once."""
        N = 3
        nc_count = {"n": 0}
        lv_count = {"n": 0}
        lock = threading.Lock()

        def make_nc_callable(val):
            def fn(val=val):
                with lock:
                    nc_count["n"] += 1
                return {"src": "nc", "val": val}

            return fn

        def make_lv_callable(val):
            def fn(val=val):
                with lock:
                    lv_count["n"] += 1
                return {"src": "lv", "val": val}

            return fn

        nc_callables = [make_nc_callable(i) for i in range(N)]
        lv_callables = [make_lv_callable(i) for i in range(N)]
        coordinator = _GroupCoordinator(nc_callables, lv_callables)
        push_group = PushGroup()

        def graph():
            nc_data = _CoordinatedPushAdapter(
                dict,
                coordinator,
                "nc",
                push_mode=csp.PushMode.NON_COLLAPSING,
                push_group=push_group,
            )
            lv_data = _CoordinatedPushAdapter(
                dict,
                coordinator,
                "lv",
                push_mode=csp.PushMode.LAST_VALUE,
                push_group=push_group,
            )
            total = csp.count(nc_data) + csp.count(lv_data)
            stop = total == 2 * N
            stop = csp.filter(stop, stop)
            csp.stop_engine(stop)
            csp.add_graph_output("nc", nc_data)
            csp.add_graph_output("lv", lv_data)

        res = _run_adapter(graph)
        nc_res = res["nc"]
        lv_res = res["lv"]

        # NC adapter: all N events should arrive
        self.assertEqual(len(nc_res), N)
        nc_values = [r[1]["val"] for r in nc_res]
        self.assertEqual(nc_values, list(range(N)))

        # LV adapter: may collapse but at least one event should arrive
        self.assertGreaterEqual(len(lv_res), 1)
        # Last LV value should be the final one pushed
        self.assertEqual(lv_res[-1][1]["val"], N - 1)

        # Each callable invoked exactly once (not double-transformed)
        self.assertEqual(nc_count["n"], N)
        self.assertEqual(lv_count["n"], N)


if __name__ == "__main__":
    unittest.main()
