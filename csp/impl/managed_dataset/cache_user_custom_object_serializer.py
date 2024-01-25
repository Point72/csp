from abc import ABCMeta, abstractmethod


class CacheObjectSerializer(metaclass=ABCMeta):
    @abstractmethod
    def serialize_to_bytes(self, value):
        raise NotImplementedError

    @abstractmethod
    def deserialize_from_bytes(self, value):
        raise NotImplementedError
