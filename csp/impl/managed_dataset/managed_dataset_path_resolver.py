import datetime
import glob
import os
from typing import Callable, Dict, List, Optional, Union

import csp
from csp.impl.constants import UNSET
from csp.impl.managed_dataset.aggregation_period_utils import AggregationPeriodUtils
from csp.impl.managed_dataset.dataset_metadata import OutputType, TimeAggregation
from csp.impl.managed_dataset.dateset_name_constants import DatasetNameConstants


class DatasetPartitionPaths:
    _FILE_EXTENSION_BY_TYPE = {OutputType.PARQUET: ".parquet"}
    _FOLDER_DATA_GLOB_EXPRESSION = (
        "[0-9]" * 8 + "_" + "[0-9]" * 6 + "_" + "[0-9]" * 6 + "-" + "[0-9]" * 8 + "_" + "[0-9]" * 6 + "_" + "[0-9]" * 6
    )

    DATA_FOLDER = "data"

    def __init__(
        self,
        dataset_root_folder: str,
        dataset_read_folders,
        partitioning_values: Dict[str, str] = None,
        time_aggregation: TimeAggregation = TimeAggregation.DAY,
    ):
        self._partition_values = tuple(partitioning_values.values())
        self._time_aggregation = time_aggregation
        if self._partition_values:
            sub_folder_parts = list(map(str, self._partition_values))
        else:
            sub_folder_parts = []

        self._root_folder = os.path.join(dataset_root_folder, self.DATA_FOLDER, *sub_folder_parts)
        self._read_folders = [os.path.join(v, self.DATA_FOLDER, *sub_folder_parts) for v in dataset_read_folders]
        self._aggregation_period_utils = AggregationPeriodUtils(time_aggregation)

    @property
    def root_folder(self):
        return self._root_folder

    @classmethod
    def _parse_file_name_times(cls, file_name):
        base_name = os.path.basename(file_name)
        start = datetime.datetime.strptime(base_name[:22], "%Y%m%d_%H%M%S_%f")
        end = datetime.datetime.strptime(base_name[23:45], "%Y%m%d_%H%M%S_%f")
        return (start, end)

    def get_period_start_time(self, start_time: datetime.datetime) -> datetime.datetime:
        """Compute the start of the period for the given timestamp
        :param start_time:
        :return:
        """
        return AggregationPeriodUtils(self._time_aggregation).resolve_period_start(start_time)

    def get_file_cutoff_time(self, start_time: datetime.datetime) -> datetime.datetime:
        """Compute the latest time that should be written to the file for which the data start at a given time
        :param start_time:
        :return:
        """
        return AggregationPeriodUtils(self._time_aggregation).resolve_period_end(start_time)

    def _get_existing_data_bound_for_root_folder(self, is_starttime, root_folder, split_columns_to_files):
        agg_bound_folder = self._aggregation_period_utils.get_agg_bound_folder(
            root_folder=root_folder, is_starttime=is_starttime
        )
        if agg_bound_folder is None:
            return None
        if split_columns_to_files:
            all_files = sorted(glob.glob(f"{glob.escape(agg_bound_folder)}/{self._FOLDER_DATA_GLOB_EXPRESSION}"))
        else:
            all_files = sorted(glob.glob(f"{glob.escape(agg_bound_folder)}/*.parquet"))
        if not all_files:
            return None
        index = 0 if is_starttime else -1
        return self._parse_file_name_times(all_files[index])[index]

    def _iterate_root_and_read_folders(self, include_root_folder=True, include_read_folders=True):
        if include_root_folder:
            yield self._root_folder
        if include_read_folders:
            for f in self._read_folders:
                yield f

    def _get_existing_data_bound_time(
        self, is_starttime, *, split_columns_to_files: bool, include_root_folder=True, include_read_folders=True
    ):
        res = None

        for root_folder in self._iterate_root_and_read_folders(
            include_root_folder=include_root_folder, include_read_folders=include_read_folders
        ):
            cur_res = self._get_existing_data_bound_for_root_folder(is_starttime, root_folder, split_columns_to_files)
            if res is None or (cur_res is not None and ((cur_res < res) == is_starttime)):
                res = cur_res
        return res

    def _normalize_start_end_time(
        self,
        starttime: datetime.datetime,
        endtime: Union[datetime.datetime, datetime.timedelta],
        split_columns_to_files: bool,
    ):
        if starttime is None:
            starttime = self._get_existing_data_bound_time(True, split_columns_to_files=split_columns_to_files)
            if starttime is None:
                return None, None

        if endtime is None:
            endtime = self._get_existing_data_bound_time(False, split_columns_to_files=split_columns_to_files)
            if endtime is None:
                return None, None
        elif isinstance(endtime, datetime.timedelta):
            endtime = starttime + endtime
        return starttime, endtime

    def _list_files_on_disk(
        self,
        starttime: datetime.datetime,
        endtime: Union[datetime.datetime, datetime.timedelta],
        split_columns_to_files=False,
        return_unused=False,
        include_read_folders=True,
    ):
        if starttime is None or endtime is None:
            return []

        files_with_times = []
        unused_files = []
        for period_start, _ in self._aggregation_period_utils.iterate_periods_in_date_range(starttime, endtime):
            file_by_base_name = {}
            for root_folder in self._iterate_root_and_read_folders(include_read_folders=include_read_folders):
                date_output_folder = self.get_output_folder_name(period_start, root_folder)
                if split_columns_to_files:
                    files = glob.glob(f"{glob.escape(date_output_folder)}/" + self._FOLDER_DATA_GLOB_EXPRESSION)
                else:
                    files = glob.glob(f"{glob.escape(date_output_folder)}/*.parquet")
                for f in files:
                    base_name = os.path.basename(f)
                    if base_name not in file_by_base_name:
                        file_by_base_name[base_name] = f
            sorted_base_names = sorted(file_by_base_name)
            files = [file_by_base_name[f] for f in sorted_base_names]

            for file in files:
                file_start, file_end = self._parse_file_name_times(file)
                # Files are sorted ascending by start_time, end_time. For a given start time, we want to keep the highest end_time
                new_record = (file_start, file_end, file)
                if files_with_times and files_with_times[-1][0] == file_start:
                    unused_files.append(files_with_times[-1][-1])
                    files_with_times[-1] = new_record
                elif files_with_times and files_with_times[-1][1] >= file_end:
                    # The file is fully included in the previous file range
                    unused_files.append(file)
                else:
                    files_with_times.append(new_record)
        return unused_files if return_unused else files_with_times

    def get_unused_files(
        self,
        starttime: datetime.datetime,
        endtime: Union[datetime.datetime, datetime.timedelta],
        split_columns_to_files=False,
    ):
        starttime, endtime = self._normalize_start_end_time(starttime, endtime, split_columns_to_files)
        return self._list_files_on_disk(
            starttime=starttime,
            endtime=endtime,
            split_columns_to_files=split_columns_to_files,
            return_unused=True,
            include_read_folders=False,
        )

    def get_data_files_in_range(
        self,
        starttime: datetime.datetime,
        endtime: Union[datetime.datetime, datetime.timedelta],
        missing_range_handler: Callable[[datetime.datetime, datetime.datetime], bool] = None,
        split_columns_to_files=False,
        truncate_data_periods=True,
        include_read_folders=True,
    ):
        """Retrieve a list of all files in the given time range (inclusive)
        :param starttime: The start time of the period
        :param endtime: The end time of the period
        :param missing_range_handler: A function that handles missing data. Will be called with (missing_period_starttime, missing_period_endtime),
        should return True, if the missing data is not an error, should return False otherwise (in which case an exception will be raised).
        By default if no missing_range_handler is specified, the function will raise exception on any missing data.
        :param split_columns_to_files: A boolean that specifies whether the columns are split into separate files
        :param truncate_data_periods: A boolean that specifies whether the time period of each file should be truncated to the period that is consumed for a given
        time range. For example consider a file that exists for period (20210101-20210201) and we pass in the starttime=20210115 and endtime=20120116 then
        for the file above the period (key of the returned dict) will be truncated to (20210115,20120116) if the flag is set to false then
        (20210101,20210201) will be returned as a key instead.
        :param include_read_folders: A boolean that specifies whether the files in "read_folders" should be included
        :returns A tuple (files, full_coverage) where data is a dictionary of period->file_path and full_coverage is a boolean that is True only
        if the whole requested period is covered by the files, False otherwise
        """
        starttime, endtime = self._normalize_start_end_time(starttime, endtime, split_columns_to_files)
        # It's a boolean but since we need to modify it from within internal function, we need to make it a list of boolean
        full_coverage = [True]

        def handle_missing_period_error_reporting(start, end):
            if not missing_range_handler or not missing_range_handler(start, end):
                raise RuntimeError(f"Missing cache data for range {start} to {end}")
            full_coverage[0] = False

        res = {}

        files_with_times = self._list_files_on_disk(
            starttime=starttime,
            endtime=endtime,
            split_columns_to_files=split_columns_to_files,
            include_read_folders=include_read_folders,
        )

        if starttime:
            for period_start, _ in self._aggregation_period_utils.iterate_periods_in_date_range(starttime, endtime):
                prev_end = None
                for file_start, file_end, file in files_with_times:
                    file_new_data_start = file_start

                    if prev_end is not None and prev_end >= file_start:
                        if file_end <= prev_end:
                            # The period of this file is fully covered in the previous one
                            continue
                        if truncate_data_periods:
                            file_new_data_start = prev_end + datetime.timedelta(microseconds=1)

                    if (
                        (starttime <= file_new_data_start <= endtime)
                        or (starttime <= file_end <= endtime)
                        or (file_new_data_start <= starttime <= endtime <= file_end)
                    ):
                        if truncate_data_periods and starttime > file_new_data_start:
                            file_new_data_start = starttime
                        if file_end > endtime and truncate_data_periods:
                            file_end = endtime
                        res[(file_new_data_start, file_end)] = file
                    prev_end = file_end

        if not res:
            if starttime is not None or endtime is not None:
                handle_missing_period_error_reporting(starttime, endtime)
            return {}, False
        else:
            ONE_MICRO = datetime.timedelta(microseconds=1)

            dict_iter = iter(res.keys())
            period_start, period_end = next(dict_iter)
            if period_start > starttime:
                handle_missing_period_error_reporting(starttime, period_start - ONE_MICRO)

            for cur_start, cur_end in dict_iter:
                if cur_start > period_end + ONE_MICRO:
                    handle_missing_period_error_reporting(period_end + ONE_MICRO, cur_start - ONE_MICRO)
                period_end = cur_end
            if period_end < endtime:
                handle_missing_period_error_reporting(period_end + ONE_MICRO, endtime)

        return res, full_coverage[0]

    def get_output_folder_name(self, start_time: Union[datetime.datetime, datetime.date], root_folder=None):
        root_folder = root_folder or self._root_folder
        return os.path.join(root_folder, self._aggregation_period_utils.get_sub_folder_name(start_time))

    def get_output_file_name(
        self,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        output_type: OutputType = OutputType.PARQUET,
        split_columns_to_files: bool = False,
    ):
        assert end_time >= start_time
        if output_type not in (OutputType.PARQUET,):
            raise NotImplementedError(f"Unsupported output type: {output_type}")

        output_folder = self.get_output_folder_name(start_time=start_time)
        assert end_time <= self._aggregation_period_utils.resolve_period_end(start_time, exclusive_end=False)
        if split_columns_to_files:
            file_extension = ""
        else:
            file_extension = self._FILE_EXTENSION_BY_TYPE[output_type]
        return os.path.join(
            output_folder,
            f"{start_time.strftime('%Y%m%d_%H%M%S_%f')}-{end_time.strftime('%Y%m%d_%H%M%S_%f')}{file_extension}",
        )


class DatasetPartitionKey:
    def __init__(self, value_dict):
        self._value_dict = value_dict
        self._key = None

    @property
    def kwargs(self):
        return self._value_dict

    def _get_key(self):
        if self._key is None:
            self._key = tuple(self._value_dict.items())
        return self._key

    def __str__(self):
        return f"DatasetPartitionKey({self._value_dict})"

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if not isinstance(other, DatasetPartitionKey):
            return False
        return self._get_key() == other._get_key()

    def __hash__(self):
        return hash(self._get_key())


class DatasetPaths(object):
    DATASET_METADATA_FILE_NAME = "dataset_meta.yml"

    def __init__(
        self,
        parent_folder: str,
        read_folders: str,
        name: str,
        time_aggregation=TimeAggregation.DAY,
        data_category: Optional[List[str]] = None,
    ):
        self._name = name
        self._time_aggregation = time_aggregation
        self._data_category = data_category

        # Note we must call the list on data_category since we want a copy that we're going to modify
        dataset_sub_folder_parts = list(data_category) if data_category else []
        dataset_sub_folder_parts.append(name)
        self._dataset_sub_folder_parts_str = os.path.join(*dataset_sub_folder_parts)
        self._parent_folder = parent_folder
        self._dataset_root_folder = os.path.abspath(os.path.join(parent_folder, self._dataset_sub_folder_parts_str))
        self._dataset_read_root_folders = (
            [os.path.abspath(os.path.join(v, *dataset_sub_folder_parts)) for v in read_folders] if read_folders else []
        )

    def get_partition_paths(self, partitioning_values: Dict[str, str] = None):
        return DatasetPartitionPaths(
            self.root_folder,
            self._dataset_read_root_folders,
            partitioning_values,
            time_aggregation=self._time_aggregation,
        )

    @property
    def parent_folder(self):
        return self._parent_folder

    @property
    def root_folder(self):
        return self._dataset_root_folder

    @classmethod
    def _get_metadata_file_path(cls, root_folder):
        return os.path.join(root_folder, cls.DATASET_METADATA_FILE_NAME)

    def get_metadata_file_path(self, existing: bool):
        """
        Get the metadata file path if "existing" is True then any metadata from either root folder or read folders will be returned (whichever exists) or None if not
        metadata file exists. If "existing" is False then the metadata for the "root_folder" will be returned, no matter if it exists or not.
        :param existing:
        :return:
        """
        if not existing:
            return os.path.join(self.root_folder, self.DATASET_METADATA_FILE_NAME)

        for folder in self._iter_root_folders(True):
            file_path = os.path.join(folder, self.DATASET_METADATA_FILE_NAME)
            if os.path.exists(file_path):
                return file_path
        return None

    def _iter_root_folders(self, use_read_folders):
        yield self._dataset_root_folder
        if use_read_folders:
            for f in self._dataset_read_root_folders:
                yield f

    def _resolve_partitions_recursively(self, metadata, cur_path, columns, column_index=0):
        if column_index >= len(columns):
            yield {}
            return

        col_name = columns[column_index]
        col_type = metadata.partition_columns[col_name]

        for sub_folder in os.listdir(cur_path):
            cur_value, sub_folder_full = self._load_value_from_path(cur_path, sub_folder, col_type)
            if cur_value is not UNSET:
                for res in self._resolve_partitions_recursively(metadata, sub_folder_full, columns, column_index + 1):
                    d = {col_name: cur_value}
                    d.update(**res)
                    yield d

    def _load_value_from_path(self, cur_path, sub_folder, col_type):
        cur_value = UNSET
        sub_folder_full = os.path.join(cur_path, sub_folder)
        if issubclass(col_type, csp.Struct):
            if os.path.isdir(sub_folder_full) and sub_folder.startswith("struct_"):
                value_file = os.path.join(sub_folder_full, DatasetNameConstants.PARTITION_ARGUMENT_FILE_NAME)
                if os.path.exists(os.path.exists(value_file)):
                    with open(value_file, "r") as f:
                        cur_value = col_type.from_yaml(f.read())
        elif col_type in (int, float, str):
            try:
                cur_value = col_type(sub_folder)
            except ValueError:
                pass
        elif col_type is datetime.date:
            try:
                cur_value = datetime.datetime.strptime(sub_folder, "%Y%m%d_000000_000000").date()
            except ValueError:
                pass
        elif col_type is datetime.datetime:
            try:
                cur_value = datetime.datetime.strptime(sub_folder, "%Y%m%d_%H%M%S_%f")
            except ValueError:
                pass
        elif col_type is datetime.timedelta:
            try:
                if sub_folder.startswith("td_") and sub_folder.endswith("us"):
                    cur_value = datetime.timedelta(microseconds=int(sub_folder[3:-2]))
            except ValueError:
                pass
        elif col_type is bool:
            if sub_folder == "True":
                cur_value = True
            elif sub_folder == "False":
                cur_value = False
        else:
            raise RuntimeError(f"Unsupported partition value type {col_type}: {sub_folder}")
        return cur_value, sub_folder_full

    def get_partition_keys(self, metadata):
        if not hasattr(metadata, "partition_columns") or not metadata.partition_columns:
            return [DatasetPartitionKey({})]

        results_set = set()
        results = []

        columns = list(metadata.partition_columns)
        for root_folder in self._iter_root_folders(True):
            data_folder = os.path.join(root_folder, DatasetPartitionPaths.DATA_FOLDER)
            for res in self._resolve_partitions_recursively(metadata, data_folder, columns=columns):
                key = DatasetPartitionKey(res)
                if key not in results_set:
                    results_set.add(key)
                    results.append(key)
        return results

    def resolve_lock_file_path(self, desired_path, use_read_folders):
        """
        :param desired_path: The desired path of the lock as if it was in the data folder (this path is modified to a separate path)
        :param use_read_folders: A boolean flags whether the read folders should be tried as the prefix for the current desired path
        :return: A tuple of (parent_folder, file_path) where parent_folder is the LAST non lock specific folder in the path (anything after this is lock specific and should
        be created with different permissions)
        """
        for f in self._iter_root_folders(use_read_folders=use_read_folders):
            if os.path.commonprefix((desired_path, f)) == f:
                parent_folder = f[: -len(self._dataset_sub_folder_parts_str)]
                rel_path = os.path.relpath(desired_path, parent_folder)
                return parent_folder, os.path.join(parent_folder, ".locks", rel_path)
        raise RuntimeError(f"Unable to resolve lock file path for file {desired_path}")
