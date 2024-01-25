import threading

from csp.impl.pushadapter import PushInputAdapter
from csp.impl.types.tstype import ts
from csp.impl.wiring import py_push_adapter_def


class _GenericPushAdapterImpl(PushInputAdapter):
    def __init__(self, generic_adapter, otype):
        self._generic_adapter = generic_adapter

    def start(self, starttime, endtime):
        self._generic_adapter._bind(self)

    def stop(self):
        self._generic_adapter._unbind()


class GenericPushAdapter:
    def __init__(self, typ: type, name: str = None):
        self._adapter = _GenericPushAdapter(self, typ)

        if name:
            self._adapter.nodedef.__name__ = name

        self._adapter_impl = None
        self._started = False
        self._stopped = False
        self._condvar = threading.Condition()

    def _bind(self, adapter_impl):
        self._adapter_impl = adapter_impl
        with self._condvar:
            self._started = True
            self._condvar.notify_all()

    def _unbind(self):
        with self._condvar:
            self._adapter_impl = None
            self._stopped = True

    def wait_for_start(self, timeout: float = None):
        with self._condvar:
            self._condvar.wait_for(lambda: self._started, timeout)

    def started(self):
        return self._started

    def stopped(self):
        return self._stopped

    # Public interface
    def out(self):
        return self._adapter

    def push_tick(self, value):
        with self._condvar:
            if self._adapter_impl is None:
                return False
            self._adapter_impl.push_tick(value)
            return True


_GenericPushAdapter = py_push_adapter_def(
    "GenericPushAdapter", _GenericPushAdapterImpl, ts["T"], generic_adapter=GenericPushAdapter, otype="T"
)
