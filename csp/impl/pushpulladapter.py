from csp.impl.__cspimpl import _cspimpl

PushGroup = _cspimpl.PushGroup
PushBatch = _cspimpl.PushBatch


class PushPullInputAdapter(_cspimpl.PyPushPullInputAdapter):
    def start(self, starttime, endtime):
        pass

    def stop(self):
        pass

    # base class
    # def push_tick( self, time, value )
