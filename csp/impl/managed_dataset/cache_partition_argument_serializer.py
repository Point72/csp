import hashlib
import io
import ruamel.yaml
from abc import ABCMeta, abstractmethod

from csp.impl.struct import Struct


class SerializedArgument:
    def __init__(self, arg, serializer):
        self._arg = arg
        self._serializer = serializer
        self._arg_as_string = None
        self._arg_as_dict = None
        self._arg_as_yaml_string = None

    def __str__(self):
        return self.arg_as_string

    @property
    def arg(self):
        return self._arg

    @property
    def arg_as_string(self):
        if self._arg_as_string is None:
            self._arg_as_string = self._serializer.to_string(self)
        return self._arg_as_string

    @property
    def arg_as_yaml_string(self):
        if self._arg_as_yaml_string is None:
            yaml = ruamel.yaml.YAML()
            string_io = io.StringIO()
            yaml.dump(self.arg_as_dict, string_io)
            self._arg_as_yaml_string = string_io.getvalue()
        return self._arg_as_yaml_string

    @property
    def arg_as_dict(self):
        if self._arg_as_dict is None:
            self._arg_as_dict = self._serializer.to_json_dict(self)
        return self._arg_as_dict


class CachePartitionArgumentSerializer(metaclass=ABCMeta):
    @abstractmethod
    def to_json_dict(self, value: SerializedArgument):
        """
        :param value: The value to serialize
        :returns: Should return a dict that will be written to yaml file
        """
        raise NotImplementedError()

    @abstractmethod
    def from_json_dict(self, value):
        """
        :param value: The dict that is read from yaml file
        :returns: Should return the deserialized object
        """
        raise NotImplementedError()

    @abstractmethod
    def to_string(self, value: SerializedArgument):
        """Serialize the given object to a string (this string will be the partition folder name)

        :param value: The value to serialize
        :returns: Should return a string that will be the folder name
        """
        raise NotImplementedError()

    def __call__(self, value):
        return SerializedArgument(value, self)


class StructPartitionArgumentSerializer(CachePartitionArgumentSerializer):
    def __init__(self, typ):
        self._typ = typ

    def to_json_dict(self, value: SerializedArgument):
        """
        :param value: The value to serialize
        :returns: Should return a dict that will be written to yaml file
        """
        assert isinstance(value.arg, self._typ)
        return value.arg.to_dict()

    def from_json_dict(self, value) -> Struct:
        """
        :param value: The dict that is read from yaml file
        :returns: Should return the deserialized object
        """
        return self._typ.from_dict(value)

    def to_string(self, value: SerializedArgument):
        """Serialize the given object to a string (this string will be the partition folder name)

        :param value: The value to serialize
        :returns: Should return a string that will be the folder name
        """
        return f"struct_{hashlib.md5(value.arg_as_yaml_string.encode()).hexdigest()}"
