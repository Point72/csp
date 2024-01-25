import unittest
from datetime import datetime, timedelta

try:
    import numba

    # We need to import csp to enable usage of datetime inside numba compiled functions
    # noinspection PyUnresolvedReferences
    import csp
    import csp.impl.wiring.numba_utils.datetime_extension

    skip_no_module = False
except ImportError:
    skip_no_module = True
import numpy


@unittest.skipIf(skip_no_module, "numa not available")
class TestNumbaDatetimeExtension(unittest.TestCase):
    EPOCH_START = datetime(1970, 1, 1)
    NUMBA_EPOCH_START = numpy.datetime64("1970-01-01")

    def _timedeltas_to_nanoseconds(self, values):
        res = []
        for v in values:
            if isinstance(v, timedelta):
                # We need to first convert to int up to micros and then
                # convert to nanos to avoid rounding problems
                res.append(int(round(v.total_seconds(), 6) * 1e6) * 1000)
            else:
                res.append(int(v))
        return res

    def test_datetime(self):
        def _create_datetimes():
            return (
                datetime(2020, 1, 2),
                datetime(year=2020, month=1, day=2),
                datetime(year=2020, month=1, day=2, hour=3),
                datetime(year=2020, month=1, day=2, minute=4),
                datetime(year=2020, month=1, day=2, second=5),
                datetime(year=2020, month=1, day=2, microsecond=6),
                datetime(year=2020, month=1, day=2, hour=3, minute=4, second=5, microsecond=6),
            )

        def _diff_datetimes(datetimes, epoch_start):
            res = []
            for d in datetimes:
                res.append(d - epoch_start)
            return res

        _numba_create_datetimes = numba.njit()(_create_datetimes)
        _numba_diff_datetimes = numba.njit()(_diff_datetimes)

        datetimes = _create_datetimes()
        numba_datetimes = _numba_create_datetimes()

        datetimes_diff = self._timedeltas_to_nanoseconds(_diff_datetimes(datetimes, self.EPOCH_START))
        numba_datetimes_diff = self._timedeltas_to_nanoseconds(
            _numba_diff_datetimes(numba_datetimes, self.NUMBA_EPOCH_START)
        )

        self.assertTrue(len(datetimes_diff) > 0)
        self.assertEqual(datetimes_diff, numba_datetimes_diff)

    def test_timedelta_creation(self):
        def _create_timedeltas():
            return (
                timedelta(),
                timedelta(1),
            )
            # Commenting these out because theyre generating a lot of numba warning noise for some reason, must be some numba bug
            """ /usr/lib/python3.8/site-packages/numba/core/ssa.py:272: NumbaIRAssumptionWarning: variable 'create_timedelta__locals__create_datetime_impl_v65_res_11.1' is not in scope.

This warning came from an internal pedantic check. Please report the warning
message and traceback, along with a minimal reproducer at:
https://github.com/numba/numba/issues/new?template=bug_report.md

  warnings.warn(errors.NumbaIRAssumptionWarning(wmsg,
 """
            """ timedelta(1, 2),
            timedelta(1, 2, 3),
            timedelta(1, 2, 3, 4),
            timedelta(1, 2, 3, 4, 5, 6),
            timedelta(1, 2, 3, 4, 5, 6, 7),
            timedelta(),
            timedelta(days=1),
            timedelta(days=1, seconds=2),
            timedelta(days=1, seconds=2, microseconds=3),
            timedelta(days=1, seconds=2, microseconds=3, milliseconds=4),
            timedelta(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6),
            timedelta(days=1, seconds=2, microseconds=3, milliseconds=4, minutes=5, hours=6, weeks=7)
            )"""

        _numba_create_timedeltas = numba.njit()(_create_timedeltas)
        time_deltas = self._timedeltas_to_nanoseconds(_create_timedeltas())
        numba_time_deltas = self._timedeltas_to_nanoseconds(_numba_create_timedeltas())

        self.assertTrue(len(time_deltas) > 0)
        self.assertEqual(time_deltas, numba_time_deltas)


if __name__ == "__main__":
    unittest.main()
