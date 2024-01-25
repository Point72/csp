from enum import IntEnum

from csp.impl.struct import Struct


class Status(Struct):
    level: int
    status_code: int
    msg: str


class Level(IntEnum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
