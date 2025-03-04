"""
An extension module that enables construction of datetime and timedelta in numba functions.
The module needs to be imported to enable the support.
It enables construction that is compatible with datetime.datetime and datetime.timedelta, the
returned types are actually numpy.datetime64 and numpy.timedelta64 in units of nanoseconds.
Regular datetime and timedelta arithmetic can be used.
"""

import datetime

import numba

from csp.impl.wiring.numba_utils.csp_cpp_numba_interface import C as csp_c

csp_numba_datetime_type = numba.types.NPDatetime("ns")
csp_numba_timedelta_type = numba.types.NPTimedelta("ns")


def _datetime_cast_implementation(ret_type, inst_type):
    """The actual implementation of the llvm casting to/from datetime and timedelta
    :param ret_type: The return type of the cast
    :param inst_type: The type of the casted value
    :return:
    """
    sig = ret_type(inst_type)

    def codegen(context, builder, signature, args):
        [src] = args
        llrtype = context.get_value_type(ret_type)
        return builder.bitcast(src, llrtype)

    return sig, codegen


@numba.extending.intrinsic
def csp_datetime_from_int(typingctx, src):
    """Numba extension function that implementes reinterpret_cast<datetime64>(int64_t)
    :param typingctx:
    :param src:
    :return:
    """
    if isinstance(src, numba.types.IntegerLiteral) or src is numba.types.int64:
        return _datetime_cast_implementation(csp_numba_datetime_type, src)
    return TypeError


@numba.extending.intrinsic
def csp_datetime_to_int(typingctx, src):
    """Numba extension function that implementes reinterpret_cast<int64_t>(datetime64)
    :param typingctx:
    :param src:
    :return:
    """

    if src is csp_numba_datetime_type:
        return _datetime_cast_implementation(numba.types.int64, src)
    return TypeError


@numba.extending.intrinsic
def csp_timedelta_from_int(typingctx, src):
    """Numba extension function that implementes reinterpret_cast<timedelta64>(int64_t)
    :param typingctx:
    :param src:
    :return:
    """
    if isinstance(src, numba.types.IntegerLiteral) or src is numba.types.int64:
        return _datetime_cast_implementation(csp_numba_timedelta_type, src)
    return TypeError


@numba.extending.intrinsic
def csp_timedelta_to_int(typingctx, src):
    """Numba extension function that implementes reinterpret_cast<int64_t>(timedelta64)
    :param typingctx:
    :param src:
    :return:
    """
    if src is csp_numba_timedelta_type:
        return _datetime_cast_implementation(numba.types.int64, src)
    return TypeError


__csp_create_datetime_nanoseconds__ = csp_c.__csp_create_datetime_nanoseconds__


@numba.extending.overload(datetime.datetime, inline="always")
def create_datetime(year, month, day, hour=0, minute=0, second=0, microsecond=0, nanosecond=0):
    """An implementation of datetime __init_ call that will produce numpy.datetime64 object instead
    :param year:
    :param month:
    :param day:
    :param hour:
    :param minute:
    :param second:
    :param microsecond:
    :param nanosecond:
    :return:
    """

    def create_datetime_impl(year, month, day, hour=0, minute=0, second=0, microsecond=0, nanosecond=0):
        return csp_datetime_from_int(
            __csp_create_datetime_nanoseconds__(year, month, day, hour, minute, second, microsecond * 1000 + nanosecond)
        )

    return create_datetime_impl


MICROS_MULT = 1000
MILLIS_MULT = MICROS_MULT * 1000
SECONDS_MULT = MILLIS_MULT * 1000
MINUTES_MULT = SECONDS_MULT * 60
HOURS_MULT = MINUTES_MULT * 60
DAYS_MULT = HOURS_MULT * 24
WEEKS_MULT = DAYS_MULT * 7


@numba.extending.overload(datetime.timedelta, inline="always")
def create_timedelta(days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0, nanoseconds=0):
    """An implementation of timedelta __init_ call that will produce numpy.timedelta64 object instead
    :param year:
    :param month:
    :param day:
    :param hour:
    :param minute:
    :param second:
    :param microsecond:
    :param nanosecond:
    :return:
    """

    def create_datetime_impl(
        days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0, nanoseconds=0
    ):
        res = nanoseconds if nanoseconds is not None else 0
        if days != 0:
            res += DAYS_MULT * days
        if seconds != 0:
            res += SECONDS_MULT * seconds
        if microseconds != 0:
            res += MICROS_MULT * microseconds
        if milliseconds != 0:
            res += MILLIS_MULT * milliseconds
        if minutes != 0:
            res += MINUTES_MULT * minutes
        if hours != 0:
            res += HOURS_MULT * hours
        if weeks != 0:
            res += WEEKS_MULT * weeks

        return csp_timedelta_from_int(res)

    return create_datetime_impl
