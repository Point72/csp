from datetime import date, datetime

import pytz
from packaging import version

try:
    import perspective

    if version.parse(perspective.__version__) >= version.parse("3"):
        _PERSPECTIVE_3 = True
        from perspective.widget import PerspectiveWidget

    elif version.parse(perspective.__version__) >= version.parse("0.6.2"):
        from perspective import PerspectiveManager, PerspectiveWidget  # noqa F401

        _PERSPECTIVE_3 = False
    else:
        raise ImportError("perspective adapter requires 0.6.2 or greater of the perspective-python package")

except ImportError:
    raise ImportError(
        "perspective must be installed to use this module. To install, run 'pip install perspective-python'."
    )


def is_perspective3():
    """Whether the perspective version is >= 3"""
    return _PERSPECTIVE_3


def perspective_type_map():
    """Return the mapping of standard python types to perspective types"""
    return {
        str: "string",
        float: "float",
        int: "integer",
        date: "date",
        datetime: "datetime",
        bool: "boolean",
    }


def datetime_to_perspective(dt: datetime) -> int:
    """Convert a python datetime to an integer number of milliseconds for perspective >= 3"""
    if dt.tzinfo is None:
        dt = pytz.utc.localize(dt)
    return int(dt.timestamp() * 1000)


def date_to_perspective(d: date) -> int:
    """Convert a python date to an integer number of milliseconds for perspective >= 3"""
    return int(datetime(year=d.year, month=d.month, day=d.day, tzinfo=pytz.UTC).timestamp() * 1000)
