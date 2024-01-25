from datetime import date, timedelta

ONE_DAY_DELTA = timedelta(days=1)


def get_dates_in_range(start: date, end: date, inclusive_end=True):
    n_days = (end - start).days + int(inclusive_end)
    return [start + ONE_DAY_DELTA * i for i in range(n_days)]
