import threading
import time
import unittest
from datetime import timedelta

import csp
from csp import ts
from csp.impl.wiring.adapters import InputAdapterDef, _adapterdef
from csp.lib import _csptestlibimpl
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

    def __init__(self, typ, callables):
        self._callables = callables
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
            time.sleep(0.01)


_CallablePushAdapter = callable_py_push_adapter_def(
    "_CallablePushAdapter", _CallablePushAdapterImpl, ts["T"], typ="T", callables=list
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


if __name__ == "__main__":
    unittest.main()
