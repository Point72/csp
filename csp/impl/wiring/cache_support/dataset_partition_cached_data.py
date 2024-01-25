import datetime
import itertools
import logging
import numpy
import os
import pytz
import shutil
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Callable, Dict, List, Optional

import csp
from csp.adapters.output_adapters.parquet import resolve_array_shape_column_name
from csp.impl.managed_dataset.aggregation_period_utils import AggregationPeriodUtils
from csp.impl.managed_dataset.dateset_name_constants import DatasetNameConstants
from csp.impl.types.typing_utils import CspTypingUtils


class DataSetCachedData:
    def __init__(self, dataset, cache_serializers, data_set_partition_calculator_func):
        self._dataset = dataset
        self._cache_serializers = cache_serializers
        self._data_set_partition_calculator_func = data_set_partition_calculator_func

    def get_partition_keys(self):
        return self._dataset.data_paths.get_partition_keys(self._dataset.metadata)

    def __call__(self, *args, **kwargs):
        return DatasetPartitionCachedData(
            self._data_set_partition_calculator_func(*args, **kwargs), self._cache_serializers
        )


class DatasetPartitionCachedData:
    def __init__(self, dataset_partition, cache_serializers):
        self._dataset_partition = dataset_partition
        self._cache_serializers = cache_serializers

    @property
    def metadata(self):
        return self._dataset_partition.dataset.metadata

    @classmethod
    def _normalize_time(cls, time: datetime.datetime, drop_tz_info=False):
        res = None
        if time is not None:
            if isinstance(time, datetime.timedelta):
                return time
            if time.tzinfo is None:
                res = pytz.utc.localize(time)
            else:
                res = time.astimezone(pytz.UTC)
        if res is not None and drop_tz_info:
            res = res.replace(tzinfo=None)
        return res

    def _get_shape_columns(self, column_list):
        for c in column_list:
            c_type = self.metadata.columns.get(c)
            if c_type and CspTypingUtils.is_numpy_nd_array_type(c_type):
                yield c, resolve_array_shape_column_name(c)

    def _get_shape_columns_dict(self, column_list):
        return dict(self._get_shape_columns(column_list))

    def _get_array_columns(self, column_list):
        for c in column_list:
            c_type = self.metadata.columns.get(c)
            if c_type and CspTypingUtils.is_numpy_array_type(c_type):
                yield c

    def _get_array_columns_set(self, column_list):
        return set(self._get_array_columns(column_list))

    def get_data_files_for_period(
        self,
        starttime: Optional[datetime.datetime] = None,
        endtime: Optional[datetime.datetime] = None,
        missing_range_handler: Callable[[datetime.datetime, datetime.datetime], bool] = None,
    ):
        """Retrieve a list of all files in the given time range (inclusive)
        :param starttime: The start time of the period
        :param endtime: The end time of the period
        :param missing_range_handler: A function that handles missing data. Will be called with (missing_period_starttime, missing_period_endtime),
        should return True, if the missing data is not an error, should return False otherwise (in which case an exception will be raised)
        """
        return self._dataset_partition.get_data_for_period(
            self._normalize_time(starttime, True), self._normalize_time(endtime, True), missing_range_handler
        )[0]

    def _truncate_df(self, starttime, endtime, df):
        import pandas

        if starttime is not None:
            starttime = pytz.UTC.localize(starttime)
        if endtime is not None:
            endtime = pytz.UTC.localize(endtime)

        timestamp_column_name = self._remove_unnamed_output_prefix(self.metadata.timestamp_column_name)

        if starttime is not None or endtime is not None:
            mask = pandas.Series(True, df.index)
            if starttime is not None:
                mask &= df[timestamp_column_name] >= starttime
            if endtime is not None:
                mask &= df[timestamp_column_name] <= endtime
            df = df[mask].reset_index(drop=True)
        return df

    @classmethod
    def _remove_unnamed_output_prefix(cls, value):
        unnamed_prefix = f"{DatasetNameConstants.UNNAMED_OUTPUT_NAME}."
        if isinstance(value, str):
            return value.replace(unnamed_prefix, "")
        else:
            value.columns = [c.replace(unnamed_prefix, "") for c in value.columns]

    def _load_single_file_all_columns(
        self, starttime, endtime, file_path, column_list, basket_column_list, struct_basket_sub_columns
    ):
        import numpy
        import pandas

        df = pandas.read_parquet(file_path, columns=column_list)
        self._remove_unnamed_output_prefix(df)
        df = self._truncate_df(starttime, endtime, df)

        shape_columns = self._get_shape_columns_dict(column_list)
        if shape_columns:
            columns_to_drop = []
            for k, v in shape_columns.items():
                df[k] = numpy.array([a.reshape(s) for a, s in zip(df[k], df[v])], dtype=object)
                columns_to_drop.append(v)
            df = df.drop(columns=columns_to_drop)
        return df

    def _create_empty_full_array(self, dtype, field_array_shape, pandas_dtype):
        import numpy

        if numpy.issubdtype(dtype, numpy.integer) or numpy.issubdtype(dtype, numpy.floating):
            field_array = numpy.full(field_array_shape, numpy.nan)
            pandas_dtype = float
        elif numpy.issubdtype(dtype, numpy.datetime64):
            field_array = numpy.full(field_array_shape, None, dtype=dtype)
        else:
            field_array = numpy.full(field_array_shape, None, dtype=object)
            pandas_dtype = object
        return field_array, pandas_dtype

    def _convert_array_columns(self, arrow_columns, column_list, array_columns, shape_columns):
        if not array_columns:
            return arrow_columns, column_list

        new_column_values = []
        new_column_list = []

        shape_columns_names = set(shape_columns.values())
        shape_column_arrays = {}
        for c, v in zip(column_list, arrow_columns):
            if c in shape_columns_names:
                shape_column_arrays[c] = numpy.array(v)

        for c, v in zip(column_list, arrow_columns):
            if c in shape_columns_names:
                continue
            if c in array_columns:
                numpy_v = numpy.array(v, dtype=object)
                shape_col_name = shape_columns.get(c)
                if shape_col_name:
                    shape_col = shape_column_arrays[shape_col_name]
                    numpy_v = numpy.array([v.reshape(shape) for v, shape in zip(numpy_v, shape_col)], dtype=object)
                new_column_values.append(numpy_v)
            else:
                new_column_values.append(v.to_pandas())
            new_column_list.append(c)
        return new_column_values, new_column_list

    def _load_data_split_to_columns(
        self, starttime, endtime, file_path, column_list, basket_column_list, struct_basket_sub_columns
    ):
        import numpy
        import pandas
        from pyarrow import Table
        from pyarrow.parquet import ParquetFile

        value_arrays = []
        for c in column_list:
            parquet_file = ParquetFile(os.path.join(file_path, f"{c}.parquet"))
            value_arrays.append(parquet_file.read().columns[0])

        array_columns = self._get_array_columns_set(column_list)
        shape_columns = self._get_shape_columns_dict(column_list)
        # If there are no array use the pyarrow table from arrays to pandas as it is faster, otherwise we need to convert columns since arrays are not
        # pyarrow native types
        if array_columns and value_arrays and value_arrays and value_arrays[0]:
            value_arrays, column_list = self._convert_array_columns(
                value_arrays, column_list, array_columns, shape_columns
            )
            res = pandas.DataFrame.from_dict(dict(zip(column_list, value_arrays)))
        else:
            res = Table.from_arrays(value_arrays, column_list).to_pandas()

        self._remove_unnamed_output_prefix(res)

        if basket_column_list:
            basket_dfs = []
            columns_l0 = list(res.columns)
            columns_l1 = [""] * len(columns_l0)

            for column in basket_column_list:
                value_type = self.metadata.dict_basket_columns[column].value_type

                if issubclass(value_type, csp.Struct):
                    columns = struct_basket_sub_columns.get(column, value_type.metadata().keys())
                    value_columns = [f"{column}.{k}" for k in columns]
                else:
                    assert (
                        column not in struct_basket_sub_columns
                    ), f"Specified sub columns for {column} but it's not a struct"
                    value_columns = [column]
                value_files = [os.path.join(file_path, f"{value_column}.parquet") for value_column in value_columns]
                symbol_file = os.path.join(file_path, f"{column}__csp_symbol.parquet")
                value_count_file = os.path.join(file_path, f"{column}__csp_value_count.parquet")
                symbol_data = ParquetFile(symbol_file).read().columns[0].to_pandas()
                value_data = [ParquetFile(value_file).read().columns[0].to_pandas() for value_file in value_files]
                value_count_data_array = ParquetFile(value_count_file).read().columns[0].to_pandas().values

                if len(value_count_data_array) == 0 or value_count_data_array[-1] == 0:
                    continue

                cycle_indices = value_count_data_array.cumsum() - 1
                value_count_indices = numpy.indices(value_count_data_array.shape)[0]
                good_index_mask = numpy.full(cycle_indices.shape, True)
                good_index_mask[1:] = cycle_indices[1:] != cycle_indices[:-1]

                index_array = numpy.full(len(symbol_data), numpy.nan)
                index_array[cycle_indices[good_index_mask]] = value_count_indices[good_index_mask]
                basked_data_index = pandas.Series(index_array).bfill().astype(int).values

                data_dict = {"index": basked_data_index, "symbol": symbol_data}
                for value_column, data in zip(value_columns, value_data):
                    data_dict[value_column] = data

                basket_data_raw = pandas.DataFrame(data_dict)
                if basket_data_raw.empty:
                    continue
                else:
                    all_symbols = basket_data_raw["symbol"].unique()
                    all_symbols.sort()
                    field_array_shape = (value_count_indices.size, all_symbols.size)
                    sym_indices = numpy.searchsorted(all_symbols, basket_data_raw.symbol.values)

                    field_matrices = {}
                    for f in value_columns:
                        pandas_dtype = basket_data_raw[f].dtype
                        dtype = basket_data_raw[f].values.dtype
                        field_array, pandas_dtype = self._create_empty_full_array(
                            dtype, field_array_shape, pandas_dtype
                        )

                        field_array[basked_data_index, sym_indices] = basket_data_raw[f]

                        field_matrices[f] = pandas.DataFrame(field_array, columns=all_symbols, dtype=pandas_dtype)

                    # pandas pivot_table is WAAAAY to slow, we have to implement our own here
                    basket_data_aligned = pandas.concat(
                        field_matrices.values(), keys=list(field_matrices.keys()), axis=1
                    )

                if column == DatasetNameConstants.UNNAMED_OUTPUT_NAME:
                    l0, l1 = zip(*basket_data_aligned.columns)
                    if issubclass(value_type, csp.Struct):
                        unnamed_prefix_len = len(DatasetNameConstants.UNNAMED_OUTPUT_NAME) + 1
                        l0 = [k[unnamed_prefix_len:] for k in l0]
                        basket_data_aligned.columns = list(zip(l0, l1))
                        columns_l0 += l0
                    else:
                        basket_data_aligned.columns = list(l1)
                        columns_l0 = None
                    columns_l1 += l1
                else:
                    columns_l0 += basket_data_aligned.columns.get_level_values(0).tolist()
                    columns_l1 += basket_data_aligned.columns.get_level_values(1).tolist()
                basket_dfs.append(basket_data_aligned)
            res = pandas.concat([res] + basket_dfs, axis=1)
            if columns_l0:
                res.columns = [columns_l0, columns_l1]

        return self._truncate_df(starttime, endtime, res)

    def _read_flat_data_from_files(self, symbol_file, value_files, num_values_to_skip, num_values_to_read):
        import pyarrow.parquet

        parquet_files = [pyarrow.parquet.ParquetFile(symbol_file)]
        if value_files:
            parquet_files += [pyarrow.parquet.ParquetFile(file) for file in value_files.values()]
        symbol_parquet_file = parquet_files[0]
        for row_group_index in range(symbol_parquet_file.num_row_groups):
            row_group = symbol_parquet_file.read_row_group(row_group_index, [])
            if num_values_to_skip >= row_group.num_rows:
                num_values_to_skip -= row_group.num_rows
                continue
            row_group_batches = [f.read_row_group(row_group_index).to_batches()[0] for f in parquet_files]
            column_names = list(itertools.chain(*(batch.schema.names for batch in row_group_batches)))
            column_values = list(itertools.chain(*(batch.columns for batch in row_group_batches)))
            row_group_table = pyarrow.Table.from_arrays(column_values, column_names)

            cur_row_group_start_index = num_values_to_skip
            num_values_to_skip = 0
            cur_row_group_num_values_to_read = int(
                min(row_group.num_rows - cur_row_group_start_index, num_values_to_read)
            )
            num_values_to_read -= int(cur_row_group_num_values_to_read)
            yield row_group_table.slice(cur_row_group_start_index, cur_row_group_num_values_to_read)
            if num_values_to_read == 0:
                return
        assert num_values_to_read == 0

    def _load_flat_basket_data(
        self,
        starttime,
        endtime,
        timestamp_file_name,
        symbol_file_name,
        value_count_file_name,
        value_files,
        need_timestamp=True,
    ):
        import numpy
        import pyarrow.parquet

        timestamps_arrow_array = pyarrow.parquet.ParquetFile(timestamp_file_name).read()[0]
        timestamps_array = numpy.array(timestamps_arrow_array)

        if timestamps_array.size == 0:
            return None
        cond = numpy.full(timestamps_array.shape, True)
        if starttime is not None:
            cond = (timestamps_array >= numpy.datetime64(starttime)) & cond
        if endtime is not None:
            cond = (timestamps_array <= numpy.datetime64(endtime)) & cond
        mask_indices = numpy.where(cond)[0]
        if mask_indices.size == 0:
            return None
        start_index, end_index = mask_indices[0], mask_indices[-1]
        value_counts = numpy.array(pyarrow.parquet.ParquetFile(value_count_file_name).read()[0])
        num_values_to_skip = value_counts[:start_index].sum()
        value_counts_sub_array = value_counts[start_index : end_index + 1]
        value_counts_sub_array_cumsum = value_counts_sub_array.cumsum()
        num_values_to_read = value_counts_sub_array_cumsum[-1] if value_counts_sub_array_cumsum.size > 0 else 0
        if num_values_to_read == 0:
            return None
        res = pyarrow.concat_tables(
            filter(
                None,
                self._read_flat_data_from_files(symbol_file_name, value_files, num_values_to_skip, num_values_to_read),
            )
        )

        if need_timestamp:
            timestamps_full = numpy.full(num_values_to_read, None, timestamps_array.dtype)
            timestamp_array_size = len(res)
            timestamps_sub_array = timestamps_array[start_index : end_index + 1]
            timestamps_sub_array = timestamps_sub_array[value_counts_sub_array != 0]
            value_counts_sub_array_cumsum_aux = value_counts_sub_array_cumsum[
                value_counts_sub_array_cumsum < timestamp_array_size
            ]
            timestamps_full[0] = timestamps_sub_array[0]
            timestamps_full[value_counts_sub_array_cumsum_aux] = timestamps_sub_array[1:]
            null_indices = numpy.where(numpy.isnat(timestamps_full))[0]
            non_null_indices = numpy.where(~numpy.isnat(timestamps_full))[0]
            fill_indices = non_null_indices[numpy.searchsorted(non_null_indices, null_indices, side="right") - 1]
            timestamps_full[null_indices] = timestamps_full[fill_indices]
            res = res.add_column(0, self.metadata.timestamp_column_name, pyarrow.array(timestamps_full))
        return res

    def _get_flat_basket_df_for_period(
        self,
        basket_field_name: str,
        symbol_column: str,
        struct_fields: List[str] = None,
        starttime: Optional[datetime.datetime] = None,
        endtime: Optional[datetime.datetime] = None,
        missing_range_handler: Callable[[datetime.datetime, datetime.datetime], bool] = None,
        num_threads=1,
        load_values=True,
        concat=True,
        need_timestamp=True,
    ):
        starttime = self._normalize_time(starttime, True)
        endtime = self._normalize_time(endtime, True)
        data_files = self.get_data_files_for_period(starttime, endtime, missing_range_handler)

        if basket_field_name is None:
            basket_field_name = DatasetNameConstants.UNNAMED_OUTPUT_NAME

        if basket_field_name not in self.metadata.dict_basket_columns:
            raise RuntimeError(f"No basket {basket_field_name} is returned from graph")

        symbol_files = [os.path.join(f, f"{basket_field_name}__csp_symbol.parquet") for f in data_files.values()]
        if load_values:
            value_type = self.metadata.dict_basket_columns[basket_field_name].value_type
            if issubclass(value_type, csp.Struct):
                struct_fields = struct_fields if struct_fields is not None else list(value_type.metadata().keys())
                value_files = [
                    {field: os.path.join(f, f"{basket_field_name}.{field}.parquet") for field in struct_fields}
                    for f in data_files.values()
                ]
            else:
                assert (
                    struct_fields is None
                ), f"Trying to provide struct_fields for non struct output {basket_field_name}"
                value_files = [
                    {basket_field_name: os.path.join(f, f"{basket_field_name}.parquet")} for f in data_files.values()
                ]
        else:
            value_files = list(itertools.repeat({}, len(symbol_files)))
        value_count_files = [
            os.path.join(f, f"{basket_field_name}__csp_value_count.parquet") for f in data_files.values()
        ]
        timestamp_files = [
            os.path.join(f, f"{self.metadata.timestamp_column_name}.parquet") for f in data_files.values()
        ]
        file_tuples = [
            (t, s, c, d) for t, s, c, d in zip(timestamp_files, symbol_files, value_count_files, value_files)
        ]

        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                tasks = [
                    pool.submit(self._load_flat_basket_data, starttime, endtime, *tup, need_timestamp=need_timestamp)
                    for tup in file_tuples
                ]
                results = list(task.result() for task in tasks)
        else:
            results = [
                self._load_flat_basket_data(starttime, endtime, *tup, need_timestamp=need_timestamp)
                for tup in file_tuples
            ]
        results = list(filter(None, results))
        if not results:
            return None
        if concat:
            import pyarrow

            return pyarrow.concat_tables(results)
        else:
            return results

    def get_flat_basket_df_for_period(
        self,
        symbol_column: str,
        basket_field_name: str = None,
        struct_fields: List[str] = None,
        starttime: Optional[datetime.datetime] = None,
        endtime: Optional[datetime.datetime] = None,
        missing_range_handler: Callable[[datetime.datetime, datetime.datetime], bool] = None,
        num_threads=1,
    ):
        res = self._get_flat_basket_df_for_period(
            basket_field_name=basket_field_name,
            symbol_column=symbol_column,
            struct_fields=struct_fields,
            starttime=starttime,
            endtime=endtime,
            missing_range_handler=missing_range_handler,
            num_threads=num_threads,
            concat=True,
        )
        if res is None:
            return None
        res_df = res.to_pandas()
        res_df.rename(columns={res_df.columns[1]: symbol_column}, inplace=True)
        res_df.columns = [self._remove_unnamed_output_prefix(c) for c in res_df.columns]
        return res_df

    def get_all_basket_ids_in_range(
        self,
        basket_field_name=None,
        starttime: Optional[datetime.datetime] = None,
        endtime: Optional[datetime.datetime] = None,
        missing_range_handler: Callable[[datetime.datetime, datetime.datetime], bool] = None,
        num_threads=1,
    ):
        import numpy

        symbol_column_name = "__csp_symbol__"
        parquet_tables = self._get_flat_basket_df_for_period(
            basket_field_name=basket_field_name,
            symbol_column=symbol_column_name,
            starttime=starttime,
            endtime=endtime,
            missing_range_handler=missing_range_handler,
            num_threads=num_threads,
            load_values=False,
            concat=False,
            need_timestamp=False,
        )
        unique_arrays = [numpy.unique(numpy.array(t[0])) for t in parquet_tables]
        return sorted(numpy.unique(numpy.concatenate(unique_arrays + unique_arrays)))

    def invalidate_cache(
        self, starttime: Optional[datetime.datetime] = None, endtime: Optional[datetime.datetime] = None
    ):
        existing_data = self.get_data_files_for_period(starttime, endtime, lambda *args, **kwargs: True)

        if not existing_data:
            return

        aggregation_period_utils = AggregationPeriodUtils(self.metadata.time_aggregation)
        if starttime is not None:
            agg_period_starttime = aggregation_period_utils.resolve_period_start(starttime)
            if starttime != agg_period_starttime:
                raise RuntimeError(
                    f"Trying to invalidate data starting on {starttime} - invalidation should be for full aggregation period (starting on {agg_period_starttime})"
                )

        if endtime is not None:
            agg_period_endtime = aggregation_period_utils.resolve_period_end(endtime, exclusive_end=False)
            if endtime != agg_period_endtime:
                raise RuntimeError(
                    f"Trying to invalidate data ending on {endtime} - invalidation should be for full aggregation period (ending on {agg_period_endtime})"
                )

        root_folders_to_possibly_remove = set()
        for k, v in existing_data.items():
            output_folder_name = self._dataset_partition.data_paths.get_output_folder_name(k[0])
            root_folders_to_possibly_remove.add(os.path.dirname(output_folder_name))
            logging.info(f"Removing {output_folder_name}")
            shutil.rmtree(output_folder_name)
        partition_root_folder = self._dataset_partition.data_paths.root_folder
        while root_folders_to_possibly_remove:
            aux = root_folders_to_possibly_remove
            root_folders_to_possibly_remove = set()
            for v in aux:
                if not v.startswith(partition_root_folder):
                    continue
                can_remove = True
                for item in os.listdir(v):
                    if not item.startswith(".") and not item.endswith("_WIP"):
                        can_remove = False
                        break
                if can_remove:
                    logging.info(f"Removing {v}")
                    shutil.rmtree(v)
                    root_folders_to_possibly_remove.add(os.path.dirname(v))

    def get_data_df_for_period(
        self,
        starttime: Optional[datetime.datetime] = None,
        endtime: Optional[datetime.datetime] = None,
        missing_range_handler: Callable[[datetime.datetime, datetime.datetime], bool] = None,
        data_loader_function: Callable[[str, Optional[List[str]]], object] = None,
        column_list=None,
        basket_column_list=None,
        struct_basket_sub_columns: Optional[Dict[str, List[str]]] = None,
        combine=True,
        num_threads=1,
    ):
        """Retrieve a list of all files in the given time range (inclusive)
        :param starttime: The start time of the period
        :param endtime: The end time of the period
        :param missing_range_handler: A function that handles missing data. Will be called with (missing_period_starttime, missing_period_endtime),
        should return True, if the missing data is not an error, should return False otherwise (in which case an exception will be raised)
        :param data_loader_function: A custom loader function that overrides the default pandas read. If non None specified, it implies combine=False. The
        function will be called with 2 arguments (file_path, column_list). The file_path is the path of the file to be loaded and column_list is the list
        of columns to be loaded (if column_list is None then all columns should be loaded)
        :param column_list: The list of columns to be loaded. If None specified then all columns will be loaded
        :param basket_column_list: The list of basket columns to be loaded. If None specified then all basket columns will be loaded.
        :param struct_basket_sub_columns: A dictionary of {basket_name: List[str]} that specifies which sub columns of the basket should be loaded. Only valid for
        struct baskets
        :param combine: Combine the loaded data frames into a single dataframe (if False, will return a list of dataframes). If data_loader_function
        is specified then combine is always treated as False
        :param num_threads: The number of threads to use for loading the data
        """
        starttime = self._normalize_time(starttime)
        endtime = self._normalize_time(endtime)
        if endtime is not None and isinstance(endtime, datetime.timedelta):
            endtime = starttime + endtime

        data_files = self.get_data_files_for_period(starttime, endtime, missing_range_handler)
        if data_loader_function is None:
            if self.metadata.split_columns_to_files:
                data_loader_function = self._load_data_split_to_columns
            else:
                data_loader_function = self._load_single_file_all_columns
        else:
            combine = False

        if column_list is None:
            column_list = list(self.metadata.columns.keys())
        shape_columns = self._get_shape_columns_dict(column_list)
        if shape_columns:
            column_list += list(shape_columns.values())

        if basket_column_list is None:
            basket_columns = getattr(self.metadata, "dict_basket_columns", None)
            basket_column_list = list(basket_columns.keys()) if basket_columns else None

        if struct_basket_sub_columns is None:
            struct_basket_sub_columns = {}
            if basket_column_list:
                for col in basket_column_list:
                    value_type = self.metadata.dict_basket_columns[col].value_type
                    if issubclass(value_type, csp.Struct):
                        self.metadata.dict_basket_columns[col]
                        struct_basket_sub_columns[col] = list(value_type.metadata().keys())
        else:
            if "" in struct_basket_sub_columns:
                struct_basket_sub_columns[DatasetNameConstants.UNNAMED_OUTPUT_NAME] = struct_basket_sub_columns.pop("")
            for k, v in struct_basket_sub_columns.items():
                if k not in basket_column_list:
                    raise RuntimeError(f"Specified sub columns for basket '{k}' but it's not loaded from file: {v}")

        if self.metadata.timestamp_column_name not in column_list:
            column_list = [self.metadata.timestamp_column_name] + column_list

        if num_threads > 1:
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                tasks = [
                    pool.submit(
                        data_loader_function,
                        file_start_time,
                        file_end_time,
                        data_file,
                        column_list,
                        basket_column_list,
                        struct_basket_sub_columns,
                    )
                    for (file_start_time, file_end_time), data_file in data_files.items()
                ]
                dfs = [task.result() for task in tasks]
        else:
            dfs = [
                data_loader_function(
                    file_start_time,
                    file_end_time,
                    data_file,
                    column_list,
                    basket_column_list,
                    struct_basket_sub_columns,
                )
                for (file_start_time, file_end_time), data_file in data_files.items()
            ]

        dfs = [df for df in dfs if len(df) > 0]

        # For now we do it in one process, in the future might push it into multiprocessing load
        for k, typ in self._dataset_partition.dataset.metadata.columns.items():
            serializer = self._cache_serializers.get(typ)
            if serializer:
                for df in dfs:
                    df[k] = df[k].apply(lambda v: serializer.deserialize_from_bytes(v) if v is not None else None)

        if combine:
            if len(dfs) > 0:
                import pandas

                return pandas.concat(dfs, ignore_index=True)
            else:
                return None
        else:
            return dfs
