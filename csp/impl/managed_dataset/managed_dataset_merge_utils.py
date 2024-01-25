import datetime
import itertools
import os
import pytz
import tempfile
from typing import Optional

import csp
from csp.adapters.output_adapters.parquet import ParquetOutputConfig, ParquetWriter
from csp.cache_support import CacheConfig
from csp.impl.managed_dataset.aggregation_period_utils import AggregationPeriodUtils
from csp.impl.managed_dataset.managed_dataset import ManagedDatasetPartition
from csp.impl.managed_dataset.managed_dataset_lock_file_util import ManagedDatasetLockUtil
from csp.utils.file_permissions import apply_file_permissions
from csp.utils.lock_file import MultipleFilesLock


def _pa():
    """
    Lazy import pyarrow
    """
    import pyarrow

    return pyarrow


def _create_wip_file(output_folder, start_time, is_folder: Optional[bool] = False):
    prefix = start_time.strftime("%Y%m%d_H%M%S_%f") if start_time else "merge_"

    if is_folder:
        return tempfile.mkdtemp(dir=output_folder, suffix="_WIP", prefix=prefix)
    else:
        fd, cur_file_path = tempfile.mkstemp(dir=output_folder, suffix="_WIP", prefix=prefix)
        os.close(fd)
        return cur_file_path


class _SingleBasketMergeData:
    def __init__(self, basket_name, basket_types, input_files, basket_data_input_files):
        self.basket_data_input_files = basket_data_input_files
        self.basket_name = basket_name
        self.basket_types = basket_types
        self.count_column_name = f"{basket_name}__csp_value_count"
        if issubclass(self.basket_types.value_type, csp.Struct):
            self.data_column_names = [f"{basket_name}.{c}" for c in self.basket_types.value_type.metadata()]
        else:
            self.data_column_names = [basket_name]
        self.symbol_column_name = f"{basket_name}__csp_symbol"
        self._cur_basket_data_row_group = None
        self._cur_row_group_data_table = None
        self._cur_row_group_symbol_table = None
        self._cur_row_group_last_returned_index = int(-1)

    def _load_row_group(self, next_row_group_index=None):
        if next_row_group_index is None:
            next_row_group_index = self._cur_basket_data_row_group + 1
        self._cur_basket_data_row_group = next_row_group_index
        do_iter = True
        while do_iter:
            if self._cur_basket_data_row_group < self.basket_data_input_files[self.data_column_names[0]].num_row_groups:
                self._cur_row_group_data_tables = [
                    self.basket_data_input_files[c].read_row_group(self._cur_basket_data_row_group)
                    for c in self.data_column_names
                ]
                self._cur_row_group_symbol_table = self.basket_data_input_files[self.symbol_column_name].read_row_group(
                    self._cur_basket_data_row_group
                )
                if self._cur_row_group_data_tables[0].shape[0] > 0:
                    do_iter = False
                else:
                    self._cur_basket_data_row_group += 1
            else:
                self._cur_row_group_data_tables = None
                self._cur_row_group_symbol_table = None
                do_iter = False
        self._cur_row_group_last_returned_index = int(-1)
        return self._cur_row_group_data_tables is not None

    @property
    def _num_remaining_rows_cur_chunk(self):
        if not self._cur_row_group_data_tables:
            return 0
        remaining_items_cur_group = (
            self._cur_row_group_data_tables[0].shape[0] - 1 - self._cur_row_group_last_returned_index
        )
        return remaining_items_cur_group

    def _skip_rows(self, num_rows_to_skip):
        remaining_items_cur_chunk = self._num_remaining_rows_cur_chunk
        while num_rows_to_skip > 0:
            if num_rows_to_skip >= remaining_items_cur_chunk:
                num_rows_to_skip -= remaining_items_cur_chunk
                assert self._load_row_group() or num_rows_to_skip == 0
            else:
                self._cur_row_group_last_returned_index += int(num_rows_to_skip)
                num_rows_to_skip = 0

    def _iter_chunks(self, row_indices, full_column_tables):
        count_table = full_column_tables[self.count_column_name].columns[0]
        count_table_cum_sum = count_table.to_pandas().cumsum()
        if self._cur_basket_data_row_group is None:
            self._load_row_group(0)

        if row_indices is None:
            if count_table_cum_sum.empty:
                return
            num_rows_to_return = int(count_table_cum_sum.iloc[-1])
        else:
            if row_indices.size == 0:
                if not count_table_cum_sum.empty:
                    self._skip_rows(count_table_cum_sum.iloc[-1])
                    return

            num_rows_to_return = int(count_table_cum_sum[row_indices[-1]])
            if row_indices[0] != 0:
                skipped_rows = int(count_table_cum_sum[row_indices[0] - 1])
                self._skip_rows(skipped_rows)
                num_rows_to_return -= skipped_rows

        while num_rows_to_return > 0:
            s_i = self._cur_row_group_last_returned_index + 1
            if num_rows_to_return < self._num_remaining_rows_cur_chunk:
                e_i = s_i + num_rows_to_return
                self._skip_rows(num_rows_to_return)
                num_rows_to_return = 0
                yield (self._cur_row_group_symbol_table[s_i:e_i],) + tuple(
                    t[s_i:e_i] for t in self._cur_row_group_data_tables
                )
            else:
                num_read_rows = self._num_remaining_rows_cur_chunk
                e_i = s_i + num_read_rows
                num_rows_to_return -= num_read_rows
                yield (self._cur_row_group_symbol_table[s_i:e_i],) + tuple(
                    t[s_i:e_i] for t in self._cur_row_group_data_tables
                )
                assert self._load_row_group() or num_rows_to_return == 0


class _MergeFileInfo(csp.Struct):
    file_path: str
    start_time: datetime.datetime
    end_time: datetime.datetime


class SinglePartitionFileMerger:
    def __init__(
        self,
        dataset_partition: ManagedDatasetPartition,
        start_time,
        end_time,
        cache_config: CacheConfig,
        parquet_output_config: ParquetOutputConfig,
    ):
        self._dataset_partition = dataset_partition
        self._start_time = start_time
        self._end_time = end_time
        self._cache_config = cache_config
        self._parquet_output_config = parquet_output_config.copy().resolve_compression()
        # TODO: cleanup all reference to existing files and backup files
        self._split_columns_to_files = getattr(dataset_partition.dataset.metadata, "split_columns_to_files", False)
        self._aggregation_period_utils = AggregationPeriodUtils(
            self._dataset_partition.dataset.metadata.time_aggregation
        )

    def _is_overwrite_allowed(self):
        allow_overwrite = getattr(self._cache_config, "allow_overwrite", None)
        if allow_overwrite is not None:
            return allow_overwrite
        allow_overwrite = getattr(self._parquet_output_config, "allow_overwrite", None)
        return bool(allow_overwrite)

    def _resolve_merged_output_file_name(self, merge_candidates):
        output_file_name = self._dataset_partition.data_paths.get_output_file_name(
            start_time=merge_candidates[0].start_time,
            end_time=merge_candidates[-1].end_time,
            split_columns_to_files=self._split_columns_to_files,
        )

        return output_file_name

    def _iterate_file_chunks(self, file_name, start_cutoff=None):
        dataset = self._dataset_partition.dataset
        parquet_file = _pa().parquet.ParquetFile(file_name)
        if start_cutoff:
            for i in range(parquet_file.metadata.num_row_groups):
                time_stamps = parquet_file.read_row_group(i, [dataset.metadata.timestamp_column_name])[
                    dataset.metadata.timestamp_column_name
                ].to_pandas()
                row_indices = time_stamps.index.values[(time_stamps > pytz.utc.localize(start_cutoff))]

                if row_indices.size == 0:
                    continue

                full_table = parquet_file.read_row_group(i)[row_indices[0] : row_indices[-1] + 1]
                yield full_table
        else:
            for i in range(parquet_file.metadata.num_row_groups):
                yield parquet_file.read_row_group(i)

    def _iter_column_names(self, include_regular_columns=True, include_basket_data_columns=True):
        dataset = self._dataset_partition.dataset
        if include_regular_columns:
            yield dataset.metadata.timestamp_column_name
            for c in dataset.metadata.columns.keys():
                yield c
        if hasattr(dataset.metadata, "dict_basket_columns"):
            for c, t in dataset.metadata.dict_basket_columns.items():
                if include_regular_columns:
                    yield f"{c}__csp_value_count"
                if include_basket_data_columns:
                    if issubclass(t.value_type, csp.Struct):
                        for field_name in t.value_type.metadata():
                            yield f"{c}.{field_name}"
                    else:
                        yield c
                    yield f"{c}__csp_symbol"

    def _iter_column_files(self, folder, include_regular_columns=True, include_basket_data_columns=True):
        for c in self._iter_column_names(
            include_regular_columns=include_regular_columns, include_basket_data_columns=include_basket_data_columns
        ):
            yield c, os.path.join(folder, f"{c}.parquet")

    def _iterate_folder_chunks(self, file_name, start_cutoff=None):
        dataset = self._dataset_partition.dataset
        input_files = {}
        for c, f in self._iter_column_files(file_name, include_basket_data_columns=False):
            input_files[c] = _pa().parquet.ParquetFile(f)

        basket_data_input_files = {}
        for c, f in self._iter_column_files(file_name, include_regular_columns=False):
            basket_data_input_files[c] = _pa().parquet.ParquetFile(f)

        timestamp_column_reader = input_files[dataset.metadata.timestamp_column_name]

        basked_data = (
            {
                k: _SingleBasketMergeData(k, v, input_files, basket_data_input_files)
                for k, v in dataset.metadata.dict_basket_columns.items()
            }
            if getattr(dataset.metadata, "dict_basket_columns", None)
            else {}
        )

        if start_cutoff:
            for i in range(timestamp_column_reader.metadata.num_row_groups):
                time_stamps = timestamp_column_reader.read_row_group(i, [dataset.metadata.timestamp_column_name])[
                    dataset.metadata.timestamp_column_name
                ].to_pandas()
                row_indices = time_stamps.index.values[(time_stamps > pytz.utc.localize(start_cutoff))]

                full_column_tables = {}
                truncated_column_tables = {}
                for c in self._iter_column_names(include_basket_data_columns=False):
                    full_table = input_files[c].read_row_group(i)
                    full_column_tables[c] = full_table
                    if row_indices.size > 0:
                        truncated_column_tables[c] = full_table[row_indices[0] : row_indices[-1] + 1]

                if row_indices.size > 0:
                    yield (
                        truncated_column_tables,
                        (
                            v._iter_chunks(row_indices=row_indices, full_column_tables=full_column_tables)
                            for v in basked_data.values()
                        ),
                    )
                else:
                    for v in basked_data.values():
                        assert (
                            len(list(v._iter_chunks(row_indices=row_indices, full_column_tables=full_column_tables)))
                            == 0
                        )
        else:
            for i in range(timestamp_column_reader.metadata.num_row_groups):
                truncated_column_tables = {}
                for c in self._iter_column_names(include_basket_data_columns=False):
                    truncated_column_tables[c] = input_files[c].read_row_group(i)
                yield (
                    truncated_column_tables,
                    (
                        v._iter_chunks(row_indices=None, full_column_tables=truncated_column_tables)
                        for v in basked_data.values()
                    ),
                )

    def _iterate_chunks(self, file_name, start_cutoff=None):
        if self._dataset_partition.dataset.metadata.split_columns_to_files:
            return self._iterate_folder_chunks(file_name, start_cutoff)
        else:
            return self._iterate_file_chunks(file_name, start_cutoff)

    def _iterate_merged_batches(self, merge_candidates):
        iters = []
        # Here we need both start time and end time to be exclusive
        start_cutoff = merge_candidates[0].start_time - datetime.timedelta(microseconds=1)
        end_cutoff = merge_candidates[-1].end_time + datetime.timedelta(microseconds=1)

        for merge_candidate in merge_candidates:
            merged_file_cutoff_start = None
            if merge_candidate.start_time <= start_cutoff:
                merged_file_cutoff_start = start_cutoff
            assert end_cutoff > merge_candidate.end_time
            iters.append(self._iterate_chunks(merge_candidate.file_path, start_cutoff=merged_file_cutoff_start))
            start_cutoff = merge_candidate.end_time
        return itertools.chain(*iters)

    def _merged_data_folders(self, aggregation_folder, merge_candidates):
        output_file_name = self._resolve_merged_output_file_name(merge_candidates)

        file_permissions = self._cache_config.data_file_permissions
        folder_permission = file_permissions.get_folder_permissions()

        wip_file = _create_wip_file(aggregation_folder, start_time=None, is_folder=True)
        apply_file_permissions(wip_file, folder_permission)
        writers = {}
        try:
            for (column1, src_file_name), (column2, file_name) in zip(
                self._iter_column_files(merge_candidates[0].file_path), self._iter_column_files(wip_file)
            ):
                assert column1 == column2
                schema = _pa().parquet.read_schema(src_file_name)
                writers[column1] = _pa().parquet.ParquetWriter(
                    file_name,
                    schema=schema,
                    compression=self._parquet_output_config.compression,
                    version=ParquetWriter.PARQUET_VERSION,
                )
            for batch, basket_batches in self._iterate_merged_batches(merge_candidates):
                for column_name, values in batch.items():
                    writers[column_name].write_table(values)

                for single_basket_column_batches in basket_batches:
                    for batch_columns in single_basket_column_batches:
                        for single_column_table in batch_columns:
                            writer = writers[single_column_table.column_names[0]]
                            writer.write_table(single_column_table)
        finally:
            for writer in writers.values():
                writer.close()

        for _, f in self._iter_column_files(wip_file):
            apply_file_permissions(f, file_permissions)

        os.rename(wip_file, output_file_name)

    def _merge_data_files(self, aggregation_folder, merge_candidates):
        output_file_name = self._resolve_merged_output_file_name(merge_candidates)

        file_permissions = self._cache_config.data_file_permissions

        wip_file = _create_wip_file(aggregation_folder, start_time=None, is_folder=False)
        schema = _pa().parquet.read_schema(merge_candidates[0].file_path)
        with _pa().parquet.ParquetWriter(
            wip_file,
            schema=schema,
            compression=self._parquet_output_config.compression,
            version=ParquetWriter.PARQUET_VERSION,
        ) as parquet_writer:
            for batch in self._iterate_merged_batches(merge_candidates):
                parquet_writer.write_table(batch)

        apply_file_permissions(wip_file, file_permissions)
        os.rename(wip_file, output_file_name)

    def _resolve_merge_candidates(self, existing_files):
        if not existing_files or len(existing_files) <= 1:
            return None

        merge_candidates = []

        for (file_period_start, file_period_end), file_path in existing_files.items():
            if not merge_candidates:
                merge_candidates.append(
                    _MergeFileInfo(file_path=file_path, start_time=file_period_start, end_time=file_period_end)
                )
                continue
            assert file_period_start >= merge_candidates[-1].start_time
            if merge_candidates[-1].end_time + datetime.timedelta(microseconds=1) >= file_period_start:
                merge_candidates.append(
                    _MergeFileInfo(file_path=file_path, start_time=file_period_start, end_time=file_period_end)
                )
            elif len(merge_candidates) <= 1:
                merge_candidates.clear()
                merge_candidates.append(
                    _MergeFileInfo(file_path=file_path, start_time=file_period_start, end_time=file_period_end)
                )
            else:
                break
        if len(merge_candidates) > 1:
            return merge_candidates
        return None

    def _merge_single_period(self, aggregation_folder, aggregation_period_start, aggregation_period_end):
        lock_file_utils = ManagedDatasetLockUtil(self._cache_config.lock_file_permissions)
        continue_merge = True
        while continue_merge:
            with lock_file_utils.merge_lock(aggregation_folder):
                existing_files, _ = self._dataset_partition.data_paths.get_data_files_in_range(
                    aggregation_period_start,
                    aggregation_period_end,
                    missing_range_handler=lambda *args, **kwargs: True,
                    split_columns_to_files=self._split_columns_to_files,
                    truncate_data_periods=False,
                    include_read_folders=False,
                )
                merge_candidates = self._resolve_merge_candidates(existing_files)
                if not merge_candidates:
                    break
                lock_file_paths = [r.file_path for r in merge_candidates]
                locks = [lock_file_utils.write_lock(f, is_lock_in_root_folder=True) for f in lock_file_paths]
                all_files_lock = MultipleFilesLock(locks)
                if not all_files_lock.lock():
                    break

            if self._dataset_partition.dataset.metadata.split_columns_to_files:
                self._merged_data_folders(aggregation_folder, merge_candidates)
            else:
                self._merge_data_files(aggregation_folder, merge_candidates)
            all_files_lock.unlock()

    def merge_files(self):
        for (
            aggregation_period_start,
            aggregation_period_end,
        ) in self._aggregation_period_utils.iterate_periods_in_date_range(
            start_time=self._start_time, end_time=self._end_time
        ):
            aggregation_period_end -= datetime.timedelta(microseconds=1)
            aggregation_folder = self._dataset_partition.data_paths.get_output_folder_name(aggregation_period_start)
            self._merge_single_period(aggregation_folder, aggregation_period_start, aggregation_period_end)
