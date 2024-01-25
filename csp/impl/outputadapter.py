from abc import ABCMeta, abstractmethod
from datetime import datetime


# NOTE: No C++ class is exported because it is not needed
# at this time. All that the engine looks for is the
# presence of a "on_tick" function
class OutputAdapter(metaclass=ABCMeta):
    @abstractmethod
    def on_tick(self, time: datetime, value: object):
        pass

    def start(self):
        pass

    def stop(self):
        pass
