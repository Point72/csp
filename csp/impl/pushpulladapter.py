from csp.impl.__cspimpl import _cspimpl
from csp.impl.error_handling import format_engine_shutdown_stack

PushGroup = _cspimpl.PushGroup
PushBatch = _cspimpl.PushBatch


class PushPullInputAdapter(_cspimpl.PyPushPullInputAdapter):
    def start(self, starttime, endtime):
        pass

    def stop(self):
        pass

    def engine_shutdown(self, msg):
        self._engine_shutdown(format_engine_shutdown_stack(msg))

    # base class
    # def push_tick( self, time, value )
