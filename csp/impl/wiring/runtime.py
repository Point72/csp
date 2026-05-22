import asyncio
import threading
import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional

import pytz

from csp.impl.__cspimpl import _cspimpl
from csp.impl.error_handling import ExceptionContext
from csp.impl.wiring.adapters import _graph_return_adapter
from csp.impl.wiring.context import Context
from csp.impl.wiring.edge import Edge
from csp.impl.wiring.outputs import OutputsContainer
from csp.profiler import Profiler, graph_info

MAX_END_TIME = datetime(2261, 12, 31, 23, 59, 50, 999999)


def _normalize_run_times(starttime, endtime, realtime):
    if starttime is None:
        if realtime:
            starttime = datetime.now(pytz.UTC).replace(tzinfo=None)
        else:
            raise RuntimeError("starttime argument is required")
    if endtime is None:
        raise RuntimeError("endtime argument is required")
    if starttime.tzinfo is not None:
        starttime = starttime.astimezone(pytz.UTC).replace(tzinfo=None)
    if isinstance(endtime, datetime) and endtime.tzinfo is not None:
        endtime = endtime.astimezone(pytz.UTC).replace(tzinfo=None)
    if isinstance(endtime, timedelta):
        endtime = starttime + endtime
    return starttime, endtime


def build_graph(f, *args, starttime=None, endtime=None, realtime=False, **kwargs):
    assert (starttime is None) == (endtime is None), (
        "Start time and end time should either both be specified or none of them should be specified when building a graph"
    )
    if starttime:
        starttime, endtime = _normalize_run_times(starttime, endtime, realtime)
    with (
        ExceptionContext(),
        GraphRunInfo(starttime=starttime, endtime=endtime, realtime=realtime),
        Context(start_time=starttime, end_time=endtime) as c,
    ):
        # Setup the profiler if within a profiling context
        if Profiler.instance() is not None and not Profiler.instance().initialized:
            Profiler.instance().init_profiler()

        outputs = f(*args, **kwargs)

        for delayed_output_node in c.delayed_nodes:
            delayed_output_node._instantiate()

        processed_outputs = OutputsContainer()

        if outputs is not None:
            if isinstance(outputs, Edge):
                processed_outputs[0] = outputs
            elif isinstance(outputs, list):
                # Unnamed list basket output
                for i, e in enumerate(outputs):
                    processed_outputs[i] = e
            elif isinstance(outputs, dict) and not isinstance(outputs, OutputsContainer):
                # Unnamed dict basket output
                for k, v in outputs.items():
                    processed_outputs[k] = v
            else:
                if not isinstance(outputs, OutputsContainer):
                    raise TypeError(
                        "graph methods need to return single Edge or OutputsContainer, got '%s'" % (type(outputs))
                    )
                for k, v in outputs._items():
                    if isinstance(v, list):
                        # Named list basket output
                        for i, item in enumerate(v):
                            processed_outputs[f"{k}[{i}]"] = item
                    elif isinstance(v, dict):
                        # Named dict basket output
                        for item_key, item in v.items():
                            processed_outputs[f"{k}[{item_key}]"] = item
                    else:
                        processed_outputs[k] = v

            for k, v in processed_outputs._items():
                _graph_return_adapter(k, v)

    if Profiler.instance() is not None:
        Profiler.instance().end_build()
    return c


def _build_engine(engine, context, memo=None):
    memo = memo or {}
    q = deque()
    for nodedef in context.roots:
        memo[nodedef] = nodedef._create(engine, memo)
        q.append(nodedef)

    while q:
        nodedef = q.popleft()
        node = memo[nodedef]

        for (idx, basket_idx), input in nodedef.ts_inputs():
            input_node = memo.get(input.nodedef, None)
            if input_node is None:
                q.append(input.nodedef)
                input_node = memo[input.nodedef] = input.nodedef._create(engine, memo)

            node.link_from(input_node, input.output_idx, input.basket_idx, idx, basket_idx)

    return engine


class GraphRunInfo:
    TLS = threading.local()

    def __init__(
        self,
        starttime,
        endtime,
        realtime,
        asyncio_on_thread=False,
        asyncio_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self._starttime = starttime
        self._endtime = endtime
        self._realtime = realtime
        # is_asyncio means same-thread asyncio execution
        # This is True when realtime=True and asyncio_on_thread=False
        self._is_asyncio = realtime and not asyncio_on_thread
        self._asyncio_loop = asyncio_loop
        self._prev = None

    @property
    def starttime(self):
        return self._starttime

    @property
    def endtime(self):
        return self._endtime

    @property
    def is_realtime(self):
        return self._realtime

    @property
    def is_asyncio(self):
        return self._is_asyncio

    @property
    def asyncio_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """Get the asyncio event loop for this run, if asyncio mode is enabled."""
        return self._asyncio_loop

    @classmethod
    def get_cur_run_times_info(cls, raise_if_missing=True):
        info = getattr(cls.TLS, "instance", None)
        if info is None and raise_if_missing:
            raise RuntimeError("csp graph information is not available outside of csp.run")
        return info

    def __enter__(self):
        self._prev = self.get_cur_run_times_info(False)
        self.TLS.instance = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prev:
            self.TLS.instance = self._prev
        else:
            del self.TLS.instance


class _WrappedContext:
    def __init__(self, context):
        self.context = context


async def _run_asyncio_engine(engine, starttime, endtime):
    """Run the engine in asyncio mode using step-based execution.

    Uses short blocking waits (1 ms) with the GIL released, interleaved
    with asyncio yields.  Push events from adapter threads wake the C++
    wait instantly via QueueWaiter::notify(), giving throughput close to
    engine.run() while still allowing asyncio coroutines to execute
    between engine cycles.

    Everything runs on the main thread so signal handling (KeyboardInterrupt,
    SIGTERM, etc.) works correctly.
    """
    wakeup_fd = engine.get_wakeup_fd()

    try:
        engine.start(starttime, endtime)

        # Let any async tasks scheduled during engine start begin.
        await asyncio.sleep(0)

        while True:
            if not engine.is_running():
                break

            # Short blocking wait (up to 1 ms).  The C++ engine releases
            # the GIL during the internal condition-variable wait so other
            # Python threads can progress.  Push events from adapter
            # threads wake it immediately via QueueWaiter::notify().
            has_more = engine.process_one_cycle(0.001)

            # Drain the wakeup pipe so it doesn't fill up.
            if wakeup_fd >= 0:
                engine.clear_wakeup_fd()

            if not has_more:
                break

            # Yield to asyncio: lets coroutines, callbacks, and scheduled
            # tasks (e.g. async adapters) execute on the main thread.
            await asyncio.sleep(0)

    except BaseException:
        engine.finish()
        raise

    return engine.finish()


def run(
    g,
    *args,
    starttime=None,
    endtime=MAX_END_TIME,
    queue_wait_time=None,
    realtime=False,
    output_numpy=False,
    asyncio_on_thread=False,
    **kwargs,
):
    # Determine if we run asyncio on the same thread (asyncio mode)
    # When realtime=True and asyncio_on_thread=False (default), we run in asyncio mode
    run_asyncio_mode = realtime and not asyncio_on_thread

    with ExceptionContext():
        starttime, endtime = _normalize_run_times(starttime, endtime, realtime)

        engine_settings = {"queue_wait_time": queue_wait_time, "realtime": realtime, "output_numpy": output_numpy}
        engine_settings = {k: v for k, v in engine_settings.items() if v is not None}

        # Wrapped context so we can release g from the calling stack and release the memory
        if isinstance(g, _WrappedContext):
            orig_g = g
            g = g.context
            orig_g.context = None

        if isinstance(g, Context):
            if g.start_time is not None:
                assert (g.start_time, g.end_time) == (starttime, endtime), (
                    f"Trying to run graph on period {(starttime, endtime)} while it was built for {(g.start_time, g.end_time)}"
                )

            if Profiler.instance() is not None:
                engine_settings["profile"] = True
                engine_settings["cycle_profile_file"] = getattr(
                    Profiler.instance(), "cycle_file", ""
                )  # optional files to write profiling data to
                engine_settings["node_profile_file"] = getattr(Profiler.instance(), "node_file", "")
                Profiler.instance().graph_info = graph_info(g).correct_from_profiler()

            engine = _cspimpl.PyEngine(**engine_settings)
            engine = _build_engine(engine, g)

            mem_cache = g.mem_cache
            # Release graph construct at this point to free up all the edge / nodedef memory thats no longer needed
            del g
            mem_cache.clear(clear_user_objects=False)

            # Ensure we dont start running realtime engines before starttime if its in the future
            now = datetime.now(pytz.UTC).replace(tzinfo=None)
            if starttime > now and realtime:
                time.sleep((starttime - now).total_seconds())

            with mem_cache:
                if run_asyncio_mode:
                    # Run in asyncio mode using step-based execution
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        # Set up GraphRunInfo with asyncio context
                        with GraphRunInfo(
                            starttime=starttime,
                            endtime=endtime,
                            realtime=realtime,
                            asyncio_on_thread=False,
                            asyncio_loop=loop,
                        ):
                            return loop.run_until_complete(_run_asyncio_engine(engine, starttime, endtime))
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)
                else:
                    return engine.run(starttime, endtime)

        if isinstance(g, Edge):
            return run(
                lambda: g, starttime=starttime, endtime=endtime, asyncio_on_thread=asyncio_on_thread, **engine_settings
            )

        # wrapped in a _WrappedContext so that we can give up the mem before run
        graph = _WrappedContext(
            build_graph(g, *args, starttime=starttime, endtime=endtime, realtime=realtime, **kwargs)
        )
        with GraphRunInfo(starttime=starttime, endtime=endtime, realtime=realtime, asyncio_on_thread=asyncio_on_thread):
            return run(
                graph, starttime=starttime, endtime=endtime, asyncio_on_thread=asyncio_on_thread, **engine_settings
            )
