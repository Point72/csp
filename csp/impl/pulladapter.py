from abc import ABCMeta, abstractmethod
from datetime import datetime


# NOTE: No C++ class is exported because it is not needed
# at this time. All that the engine looks for is the
# presence of a "next" generator, vs the Push and Output
# adapters require some additional method definitions
# in C++
class PullInputAdapter(metaclass=ABCMeta):
    __slots__ = ["_start_time", "_end_time"]

    def __init__(self):
        self._start_time = datetime.min
        self._end_time = datetime.max

    def start(self, start_time: datetime, end_time: datetime):
        """called when input adapter is meant to start ( optional )"""
        self._start_time = start_time
        self._end_time = end_time

    def stop(self):
        """called when input adapter should stop ( optional )"""
        pass

    @abstractmethod
    def next(self):
        """should return None if no more data is available, or a tuple of ( datetime, value ) of the next tick"""
        pass
