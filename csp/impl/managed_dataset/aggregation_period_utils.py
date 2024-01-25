import datetime
import glob
import os

from csp.impl.managed_dataset.dataset_metadata import TimeAggregation


class AggregationPeriodUtils:
    _AGG_LEVELS_GLOB_EXPRESSIONS = {
        TimeAggregation.DAY: ["[0-9]" * 4, "[0-9]" * 2, "[0-9]" * 2],
        TimeAggregation.MONTH: ["[0-9]" * 4, "[0-9]" * 2],
        TimeAggregation.QUARTER: ["[0-9]" * 4, "Q[0-9]"],
        TimeAggregation.YEAR: ["[0-9]" * 4],
    }

    def __init__(self, aggregation_period: TimeAggregation):
        self._aggregation_period = aggregation_period

    def resolve_period_start(self, cur_time: datetime.datetime):
        if self._aggregation_period == TimeAggregation.DAY:
            return datetime.datetime(cur_time.year, cur_time.month, cur_time.day)
        elif self._aggregation_period == TimeAggregation.MONTH:
            return datetime.datetime(cur_time.year, cur_time.month, 1)
        elif self._aggregation_period == TimeAggregation.QUARTER:
            return datetime.datetime(cur_time.year, ((cur_time.month - 1) // 3) * 3 + 1, 1)
        elif self._aggregation_period == TimeAggregation.YEAR:
            return datetime.datetime(cur_time.year, 1, 1)
        else:
            raise RuntimeError(f"Unsupported aggregation period {self._aggregation_period}")

    def resolve_period_end(self, cur_time: datetime.datetime, exclusive_end=True):
        if self._aggregation_period == TimeAggregation.DAY:
            res = datetime.datetime(cur_time.year, cur_time.month, cur_time.day) + datetime.timedelta(days=1)
        elif self._aggregation_period == TimeAggregation.MONTH:
            next_month_date = cur_time + datetime.timedelta(days=32 - cur_time.day)
            res = datetime.datetime(next_month_date.year, next_month_date.month, 1)
        elif self._aggregation_period == TimeAggregation.QUARTER:
            extra_months = (3 - cur_time.month) % 3
            next_quarter_date = cur_time + datetime.timedelta(days=31 * extra_months + 32 - cur_time.day)
            res = datetime.datetime(next_quarter_date.year, next_quarter_date.month, 1)
        elif self._aggregation_period == TimeAggregation.YEAR:
            res = datetime.datetime(cur_time.year + 1, 1, 1)
        else:
            raise RuntimeError(f"Unsupported aggregation period {self._aggregation_period}")
        if not exclusive_end:
            res -= datetime.timedelta(microseconds=1)
        return res

    def resolve_period_start_end(self, cur_time: datetime.datetime, exclusive_end=True):
        return self.resolve_period_start(cur_time), self.resolve_period_end(cur_time, exclusive_end=exclusive_end)

    def get_sub_folder_name(self, cur_time: datetime.datetime):
        if self._aggregation_period == TimeAggregation.DAY:
            return cur_time.strftime("%Y/%m/%d")
        elif self._aggregation_period == TimeAggregation.MONTH:
            return cur_time.strftime("%Y/%m")
        elif self._aggregation_period == TimeAggregation.QUARTER:
            quarter_index = (cur_time.month - 1) // 3 + 1
            return cur_time.strftime(f"%Y/Q{quarter_index}")
        elif self._aggregation_period == TimeAggregation.YEAR:
            return cur_time.strftime("%Y")
        else:
            raise RuntimeError(f"Unsupported aggregation period {self._aggregation_period}")

    def iterate_periods_in_date_range(self, start_time: datetime.datetime, end_time: datetime.datetime):
        assert start_time <= end_time
        period_start, period_end = self.resolve_period_start_end(start_time)
        while period_start <= end_time:
            yield period_start, period_end
            period_start = period_end
            period_end = self.resolve_period_end(period_start)

    def get_agg_bound_folder(self, root_folder: str, is_starttime: bool):
        """Return the first/last partition folder for the dataset
        :param root_folder:
        :param is_starttime:
        :return:
        """
        glob_expressions = self._AGG_LEVELS_GLOB_EXPRESSIONS[self._aggregation_period]
        cur = root_folder
        ind = 0 if is_starttime else -1
        for glob_exp in glob_expressions:
            cur_list = sorted(glob.glob(os.path.join(glob.escape(cur), glob_exp)))
            if not cur_list:
                return None
            cur = os.path.join(cur, cur_list[ind])
        return cur
