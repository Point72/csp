import csp
from enum import IntEnum

class Status( csp.Struct ):
    level: int
    status_code: int
    msg: str

class Level( IntEnum ):
    DEBUG    = 0
    INFO     = 1
    WARNING  = 2
    ERROR    = 3
    CRITICAL = 4