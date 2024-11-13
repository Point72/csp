from enum import IntEnum


# Must match c++ enum
class DateTimeType(IntEnum):
    UNKNOWN = 0
    UINT64_NANOS = 1
    UINT64_MICROS = 2
    UINT64_MILLIS = 3
    UINT64_SECONDS = 4


class MsgMapper:
    def __init__(self, msg_type, protocol):
        self.properties = {"msg_type": msg_type, "protocol": protocol}


class BytesMessageProtoMapper(MsgMapper):
    def __init__(self, proto_directory, proto_filename, proto_message):
        super().__init__("BYTES_MSG", "PROTOBUF")
        self.properties["proto_directory"] = proto_directory
        self.properties["proto_filename"] = proto_filename
        self.properties["proto_message"] = proto_message


class JSONTextMessageMapper(MsgMapper):
    def __init__(self, datetime_type=DateTimeType.UINT64_NANOS):
        super().__init__("TEXT_MSG", "JSON")
        self.properties["datetime_type"] = datetime_type.name


class RawTextMessageMapper(MsgMapper):
    def __init__(self):
        super().__init__("TEXT_MSG", "RAW_BYTES")


class RawBytesMessageMapper(MsgMapper):
    def __init__(self):
        super().__init__("BYTES_MSG", "RAW_BYTES")


def hash_mutable(obj):
    if isinstance(obj, (list, tuple, set)):
        return hash(tuple(hash_mutable(x) for x in obj))
    elif isinstance(obj, dict):
        return hash(tuple((hash(k), hash_mutable(v)) for k, v in obj.items()))
    else:
        return hash(obj)
