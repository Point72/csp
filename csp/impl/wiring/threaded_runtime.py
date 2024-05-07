import threading

import csp
from csp.impl.pushadapter import PushInputAdapter
from csp.impl.types.tstype import ts
from csp.impl.wiring import MAX_END_TIME, py_push_adapter_def

__all__ = ["run_on_thread"]


class _EngineStopSignalImpl(PushInputAdapter):
    def __init__(self, pusher):
        self._pusher = pusher

    def start(self, starttime, endtime):
        self._pusher.register_impl(self)

    def stop(self):
        self._pusher.unregister_impl()

    def signal_stop(self):
        self.push_tick(True)


_EngineStopSignal = py_push_adapter_def("_EngineStopSignal", _EngineStopSignalImpl, ts[bool], pusher=object)


class _EngineStopper:
    def __init__(self):
        self._pushimpl = None
        self._deferred_stop = False

    def register_impl(self, impl):
        self._pushimpl = impl
        # For the case where stop is called before push input was actually created on the engine thread
        if self._deferred_stop:
            self._pushimpl.signal_stop()

    def unregister_impl(self):
        self._pushimpl = None

    def stop_engine(self):
        if self._pushimpl is not None:
            self._pushimpl.signal_stop()
        else:
            self._deferred_stop = True


class ThreadRunner:
    def __init__(self, g, *args, **kwargs):
        self._graph = g
        self._args = args
        self._kwargs = kwargs
        self._auto_shutdown = kwargs.pop("auto_shutdown", False)
        self._stopper = _EngineStopper()
        self._output = {}
        self._thread = threading.Thread(
            target=self._run,
            args=(self._output, self._graph, self._stopper) + args,
            kwargs=self._kwargs,
            daemon=kwargs.pop("daemon", None),
        )
        self._thread.start()

    # Note: For the auto-shutdown option to work properly, the thread cannot hold a reference back to the ThreadRunner
    # (otherwise it will never be garbage collected). Thus, the _run method must be static and must only be passed
    # variables from self rather than "self". To capture output, we send in a dictionary that the _run method will
    # modify on completion.

    @staticmethod
    def _wrapped_graph(graph, stopper, *args, **kwargs):
        stop_signal = _EngineStopSignal(stopper)
        csp.stop_engine(stop_signal)
        return graph(*args, **kwargs)

    @staticmethod
    def _run(output, graph, stopper, *args, **kwargs):
        try:
            output["result"] = csp.run(ThreadRunner._wrapped_graph, graph, stopper, *args, **kwargs)
        except Exception as e:
            output["exc_info"] = e

    def stop_engine(self):
        """request engine to stop ( async )"""
        if self._stopper:
            self._stopper.stop_engine()

    def join(self, suppress=False):
        """wait for engine thread to finish and return results. If suppress=True, will suppress exceptions."""
        self._thread.join()
        if self._output.get("exc_info") and not suppress:
            raise RuntimeError from self._output["exc_info"]

        return self._output.get("result")

    def is_alive(self):
        """Checks whether the thread is still running"""
        return self._thread.is_alive()

    def __del__(self):
        if self._auto_shutdown:
            self.stop_engine()
            self.join(suppress=True)


def run_on_thread(
    g,
    *args,
    starttime=None,
    endtime=MAX_END_TIME,
    queue_wait_time=None,
    realtime=False,
    auto_shutdown=False,
    daemon=False,
    **kwargs,
):
    return ThreadRunner(
        g,
        *args,
        starttime=starttime,
        endtime=endtime,
        queue_wait_time=queue_wait_time,
        realtime=realtime,
        auto_shutdown=auto_shutdown,
        daemon=daemon,
        **kwargs,
    )
