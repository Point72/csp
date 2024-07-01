import traceback
from csp.impl.__cspimpl import _cspimpl

PushGroup = _cspimpl.PushGroup
PushBatch = _cspimpl.PushBatch


class PushInputAdapter(_cspimpl.PyPushInputAdapter):
    def start(self, starttime, endtime):
        pass

    def stop(self):
        pass

    def engine_shutdown(self, exc):
        self._engine_shutdown(''.join(traceback.format_exception(exc)).strip())

    # base class
    # def push_tick( self, value )
