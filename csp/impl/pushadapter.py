from csp.impl.__cspimpl import _cspimpl

PushGroup = _cspimpl.PushGroup
PushBatch = _cspimpl.PushBatch


class PushInputAdapter(_cspimpl.PyPushInputAdapter):
    """Base class for adapters that push ticks into a running engine from an external thread.

    Whatever owns the producer -- this adapter, or the `AdapterManager` driving it -- must quiesce
    it before engine teardown completes, and that covers every path that reaches the engine, not
    just `push_tick()`: an outstanding `PushBatch` also schedules when it is destroyed. Quiesce
    either by joining the producer thread or by gating the push behind the same lock that stop
    takes (see `GenericPushAdapter`). A tick that lands after teardown writes to a destroyed event
    queue and to a wakeup descriptor that may already have been closed and recycled.

    Note `Engine::stop()` stops input adapters before adapter managers, so a manager-owned
    producer is legitimately still alive when this adapter's `stop()` returns.
    """

    def start(self, starttime, endtime):
        pass

    def stop(self):
        pass

    # base class
    # def push_tick( self, value )
