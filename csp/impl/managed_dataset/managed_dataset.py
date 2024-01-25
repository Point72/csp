import logging
import os
import tempfile
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import csp
from csp.impl.config import BaseCacheConfig
from csp.impl.enum import Enum
from csp.impl.managed_dataset.cache_partition_argument_serializer import (
    SerializedArgument,
    StructPartitionArgumentSerializer,
)
from csp.impl.managed_dataset.dataset_metadata import DatasetMetadata, DictBasketInfo, TimeAggregation
from csp.impl.managed_dataset.dateset_name_constants import DatasetNameConstants
from csp.impl.managed_dataset.managed_dataset_lock_file_util import LockContext, ManagedDatasetLockUtil
from csp.impl.managed_dataset.managed_dataset_path_resolver import DatasetPaths
from csp.impl.struct import Struct
from csp.utils.file_permissions import FilePermissions, apply_file_permissions, create_folder_with_permissions
from csp.utils.rm_utils import rm_file_or_folder


class _MetadataRWUtil:
    def __init__(self, dataset, metadata_file_path, metadata, lock_file_permissions, data_file_permissions):
        self._dataset = dataset
        self._metadata = metadata
        self._metadata_file_path = metadata_file_path
        self._lock_file_util = ManagedDatasetLockUtil(lock_file_permissions)
        self._data_file_permissions = data_file_permissions

    def _write_metadata(self):
        locked_folder = os.path.dirname(self._metadata_file_path)
        with self._lock_file_util.write_lock(locked_folder):
            if os.path.exists(self._metadata_file_path):
                return

            file_base_name = os.path.basename(self._metadata_file_path)
            create_folder_with_permissions(locked_folder, self._dataset.cache_config.data_file_permissions)
            with tempfile.NamedTemporaryFile(mode="w+", prefix=file_base_name, dir=locked_folder, delete=False) as f:
                try:
                    yaml = self._metadata.to_yaml()
                    f.file.write(yaml)
                    f.file.flush()
                    apply_file_permissions(f.name, self._data_file_permissions)
                    os.rename(f.name, self._metadata_file_path)
                except:
                    rm_file_or_folder(f.name)
                    raise

    def load_existing_or_store_metadata(self):
        """Loads existing metadata if no metadata exists, also will store the current metadata to file

        :return: A tuple of (loaded_metadata, file_lock) where file lock locks the metadata folder for reading (the file lock is in acquired state)
        """
        if not os.path.exists(self._metadata_file_path):
            self._write_metadata()

        locked_folder = os.path.dirname(self._metadata_file_path)
        read_lock = self._lock_file_util.read_lock(locked_folder)
        read_lock.lock()
        try:
            with open(self._metadata_file_path, "r") as f:
                existing_metadata = self._metadata.from_yaml(f.read())
            return existing_metadata, read_lock
        except:
            read_lock.unlock()
            raise


class ManagedDatasetPartition:
    """A single partition of a dataset, this basically represents the lowest level of the chain dataset->partition.

    Single partition corresponds to a single instance of partition values. For example if there is a dataset that is partitioned on columns
    a:int, b:float, c:str then a single partition would for example correspond to (1, 1.0, 'str1') while (2,2.0, 'str2') would be a different
    partition object.
    Single partition corresponds to a single instance of graph.
    """

    # In the future we are going to support containers as well, for now just primitives
    PARTITION_TYPE_STR_CONVERTORS = {
        bool: str,
        int: str,
        float: str,
        str: str,
        datetime: lambda v: v.strftime("%Y%m%d_%H%M%S_%f"),
        date: lambda v: v.strftime("%Y%m%d_%H%M%S_%f"),
        timedelta: lambda v: f"td_{int(v.total_seconds() * 1e6)}us",
    }

    def __init__(self, dataset, partition_values: Optional[Dict[str, object]] = None):
        """
        :param dataset: An instance of ManagedDataset to which the partition belongs
        :param partition_values: A dictionary of partition column name to value of the column for the given partition
        """
        self._dataset = dataset
        self._values_tuple, self._values_dict = self._normalize_partition_values(partition_values)
        self._data_paths = None

    def get_data_for_period(self, starttime: datetime, endtime: datetime, missing_range_handler):
        return self.data_paths.get_data_files_in_range(
            starttime,
            endtime,
            missing_range_handler=missing_range_handler,
            split_columns_to_files=self.dataset.metadata.split_columns_to_files,
        )

    @property
    def data_paths(self):
        if self._data_paths is None:
            dataset_data_paths = self._dataset.data_paths
            if dataset_data_paths:
                self._data_paths = dataset_data_paths.get_partition_paths(self._values_dict)
        return self._data_paths

    @property
    def dataset(self):
        return self._dataset

    @property
    def value_tuple(self):
        return self._values_tuple

    @property
    def value_dict(self):
        return self._values_dict

    def _create_folder_with_permissions(self, cur_root_path, folder_permissions):
        if not os.path.exists(cur_root_path):
            try:
                os.mkdir(cur_root_path)
                apply_file_permissions(cur_root_path, folder_permissions)
                return True
            except FileExistsError:
                pass
        return False

    def create_root_folder(self, cache_config):
        if os.path.exists(self.data_paths.root_folder):
            return
        cur_root_path = self.dataset.data_paths.root_folder
        rel_path = os.path.relpath(self.data_paths.root_folder, cur_root_path)
        path_parts = list(filter(None, os.path.normpath(rel_path).split(os.sep)))
        assert path_parts[0] == "data"
        cur_root_path = os.path.join(cur_root_path, path_parts[0])

        file_permissions = cache_config.data_file_permissions
        folder_permissions = cache_config.data_file_permissions.get_folder_permissions()

        self._create_folder_with_permissions(cur_root_path, folder_permissions)
        values_dict = self._values_dict
        assert len(values_dict) + 1 == len(path_parts)
        lock_util = ManagedDatasetLockUtil(cache_config.lock_file_permissions)

        with self.dataset.use_lock_context():
            for sub_folder, argument_value in zip(path_parts[1:], values_dict.values()):
                cur_root_path = os.path.join(cur_root_path, sub_folder)
                self._create_folder_with_permissions(cur_root_path, folder_permissions)
                if isinstance(argument_value, SerializedArgument):
                    value_file_path = os.path.join(cur_root_path, DatasetNameConstants.PARTITION_ARGUMENT_FILE_NAME)
                    if not os.path.exists(value_file_path):
                        with lock_util.write_lock(value_file_path, is_lock_in_root_folder=True) as lock_file:
                            if not os.path.exists(value_file_path):
                                with open(value_file_path, "w") as value_file:
                                    value_file.write(argument_value.arg_as_yaml_string)
                                apply_file_permissions(value_file_path, file_permissions)
                            rm_file_or_folder(lock_file.file_path, is_file=True)

    def publish_file(self, file_name, start_time, end_time, file_permissions=None, lock_file_permissions=None):
        output_file_name = self.data_paths.get_output_file_name(
            start_time, end_time, split_columns_to_files=self.dataset.metadata.split_columns_to_files
        )
        # We might try to publish some files that are already there. Example
        # We ran 20210101-20210102. We now run 20210102-20210103, since the data is not fully in cache we will run the graph, the data for 20210102 will be generated again.
        if os.path.exists(output_file_name):
            rm_file_or_folder(file_name)
            return

        if file_permissions is not None:
            if os.path.isdir(file_name):
                folder_permissions = file_permissions.get_folder_permissions()
                apply_file_permissions(file_name, folder_permissions)
                for f in os.listdir(file_name):
                    apply_file_permissions(os.path.join(file_name, f), file_permissions)
            else:
                apply_file_permissions(file_name, file_permissions)

        with self.dataset.use_lock_context():
            lock_util = ManagedDatasetLockUtil(lock_file_permissions)
            with lock_util.write_lock(output_file_name, is_lock_in_root_folder=True) as lock:
                if os.path.exists(output_file_name):
                    logging.warning(f"Not publishing {output_file_name} since it already exists")
                    rm_file_or_folder(file_name)
                else:
                    os.rename(file_name, output_file_name)
                lock.delete_file()

    def merge_files(self, start_time: datetime, end_time: datetime, cache_config, parquet_output_config):
        from csp.impl.managed_dataset.managed_dataset_merge_utils import SinglePartitionFileMerger

        with self.dataset.use_lock_context():
            file_merger = SinglePartitionFileMerger(
                dataset_partition=self,
                start_time=start_time,
                end_time=end_time,
                cache_config=cache_config,
                parquet_output_config=parquet_output_config,
            )
            file_merger.merge_files()

    def cleanup_unneeded_files(self, start_time: datetime, end_time: datetime, cache_config):
        unused_files = self.data_paths.get_unused_files(
            starttime=start_time, endtime=end_time, split_columns_to_files=self.dataset.metadata.split_columns_to_files
        )
        if unused_files:
            with self.dataset.use_lock_context():
                lock_util = ManagedDatasetLockUtil(cache_config.lock_file_permissions)
                for f in unused_files:
                    try:
                        with lock_util.write_lock(
                            f, is_lock_in_root_folder=True, timeout_seconds=0, retry_period_seconds=0
                        ) as lock:
                            rm_file_or_folder(f)
                            lock.delete_file()
                    except BlockingIOError:
                        logging.warning(f"Not removing {f} since it's currently locked")

    def partition_merge_lock(self, start_time: datetime, end_time: datetime):
        raise NotImplementedError()

    def _get_type_convertor(self, typ):
        if issubclass(typ, Enum):
            return str
        elif issubclass(typ, Struct):
            return StructPartitionArgumentSerializer(typ)
        else:
            return self.PARTITION_TYPE_STR_CONVERTORS[typ]

    def _normalize_partition_values(self, partition_values):
        metadata = self._dataset.metadata
        if partition_values:
            assert len(partition_values) == len(metadata.partition_columns)
            assert partition_values.keys() == metadata.partition_columns.keys()
            ordered_partition_values = ((k, partition_values[k]) for k in metadata.partition_columns)
            partition_values = {k: self._get_type_convertor(type(v))(v) for k, v in ordered_partition_values}
            values_tuple = tuple(partition_values.values())
        else:
            assert not hasattr(metadata, "partition_columns")
            values_tuple = tuple()
        return values_tuple, partition_values


class ManagedDataset:
    """A single dataset, this basically represents the highest level of the chain dataset->partition.

    Single dataset corresponds to a set of dataset_partitions all having identical schema but having different partition keys.
    Example consider having cached trades for each ticker and date. Single dataset represents all the "trades" and has the trade
    schema attached to it. Each partition will be part of the dataset but correspond to a different ticker.
    Single dataset corresponds to a single "graph" function (defines paths and schemas for all instances of this graph).
    """

    SUPPORTED_PARTITION_TYPES = set(ManagedDatasetPartition.PARTITION_TYPE_STR_CONVERTORS.keys())

    def __init__(
        self,
        name,
        category: List[str] = None,
        timestamp_column_name: str = None,
        columns_types: Dict[str, object] = None,
        partition_columns: Dict[str, type] = None,
        *,
        cache_config: BaseCacheConfig,
        split_columns_to_files: Optional[bool],
        time_aggregation: TimeAggregation,
        dict_basket_column_types: Dict[str, Union[Tuple[type, type], DictBasketInfo]] = None,
    ):
        """
        :param name: The name of the dataset:
        :param category: The category classification of the dataset, for example ['stats', 'daily'], or ['forecasts'],
        this is being used as part of the path of the dataset on disk
         :param timestamp_column_name: The name of the timestamp column in the parquet files.
        :param columns_types: A dictionary of name->type of dataset column types.
        :param partition_columns: A dictionary of partitioning columns of the dataset. This columns are not written into parquet files but instead
            are used as part of the dataset partition path.
        :param cache_config: The cache configuration for the data set
        :param split_columns_to_files: A boolean that specifies whether the data of the dataset is split across files.
        :param time_aggregation: The data aggregation period for the dataset
        :param dict_basket_column_types: The dictionary basket columns of the dataset
        """
        self._name = name
        self._category = category if category else []
        self._cache_config = cache_config
        self._lock_context = None
        self._metadata = DatasetMetadata(
            name=name,
            split_columns_to_files=True if split_columns_to_files else False,
            time_aggregation=time_aggregation,
        )
        dict_basket_columns = self._normalize_dict_basket_types(dict_basket_column_types)
        if dict_basket_columns:
            self._metadata.dict_basket_columns = dict_basket_columns
        if timestamp_column_name:
            self._metadata.timestamp_column_name = timestamp_column_name
        self._metadata.columns = columns_types if columns_types else {}
        if partition_columns:
            self._metadata.partition_columns = partition_columns

        self._data_paths: Optional[DatasetPaths] = None
        self._set_folders(cache_config.data_folder, getattr(cache_config, "read_folders", None))

    @classmethod
    def _normalize_dict_basket_types(cls, dict_basket_column_types):
        if not dict_basket_column_types:
            return None
        dict_types = {}
        for name, type_entry in dict_basket_column_types.items():
            if isinstance(type_entry, DictBasketInfo):
                dict_types[name] = type_entry
            else:
                key_type, value_type = type_entry
                dict_types[name] = DictBasketInfo(key_type=key_type, value_type=value_type)
        return dict_types

    @classmethod
    def load_from_disk(cls, cache_config, name, data_category: Optional[List[str]] = None):
        data_paths = DatasetPaths(
            parent_folder=cache_config.data_folder,
            read_folders=getattr(cache_config, "read_folders", None),
            name=name,
            data_category=data_category,
        )
        metadata_file_path = data_paths.get_metadata_file_path(existing=True)
        if metadata_file_path:
            with open(metadata_file_path, "r") as f:
                metadata = DatasetMetadata.from_yaml(f.read())
            res = ManagedDataset(
                name=metadata.name,
                category=data_category,
                timestamp_column_name=metadata.timestamp_column_name,
                columns_types=metadata.columns,
                cache_config=cache_config,
                split_columns_to_files=metadata.split_columns_to_files,
                time_aggregation=metadata.time_aggregation,
                dict_basket_column_types=getattr(metadata, "dict_basket_columns", None),
            )
            if hasattr(metadata, "partition_columns"):
                res.metadata.partition_columns = metadata.partition_columns
            return res
        else:
            return None

    @classmethod
    def is_supported_partition_type(cls, typ):
        if typ in ManagedDataset.SUPPORTED_PARTITION_TYPES or (
            isinstance(typ, type) and (issubclass(typ, Enum) or issubclass(typ, Struct))
        ):
            return True
        else:
            return False

    @property
    def cache_config(self):
        assert self._cache_config is not None
        return self._cache_config

    def use_lock_context(self):
        if self._lock_context is None:
            self._lock_context = LockContext(self)
        return ManagedDatasetLockUtil.set_dataset_context(self._lock_context)

    def validate_and_lock_metadata(
        self,
        lock_file_permissions: Optional[FilePermissions] = None,
        data_file_permissions: Optional[FilePermissions] = None,
        read: bool = False,
        write: bool = False,
    ):
        """Validate that code metadata correspond to existing metadata on disk. If necessary writes metadata file to disk.

        :param lock_file_permissions: The permissions of the lock files that are created for safely accessing metadata.
        :param data_file_permissions: The permissions of the written metadata files.
        :param read: A bool that specifies whether the dataset will be read
        :param write: A bool that specifies whether the dataset will be written.

        Note: validation for read vs written datasets will be different. For read datasets we allow slightly different more relaxed schemas.

        :return: An obtained "shared" lock that locks the dataset schema. Caller is responsible for releasing the lock
        """
        assert self.data_paths is not None
        with self.use_lock_context():
            metadata = self.metadata
            metadata_file_path = self.data_paths.get_metadata_file_path(existing=not write)
            metadata_rw_util = _MetadataRWUtil(
                dataset=self,
                metadata_file_path=metadata_file_path,
                metadata=self.metadata,
                lock_file_permissions=lock_file_permissions,
                data_file_permissions=data_file_permissions,
            )
            existing_meta, read_lock = metadata_rw_util.load_existing_or_store_metadata()
            is_metadata_different = False

            if write:
                is_metadata_different = existing_meta != self.metadata
            else:
                for field in DatasetMetadata.metadata():
                    if field not in ("columns", "dict_basket_columns"):
                        if getattr(existing_meta, field, None) != getattr(self.metadata, field, None):
                            is_metadata_different = True
                # The read metadata must be a subset of the existing metadata
                existing_meta_columns = existing_meta.columns
                existing_meta_dict_columns = getattr(existing_meta, "dict_basket_columns", None)
                for col_name, col_type in metadata.columns.items():
                    existing_type = existing_meta_columns.get(col_name, None)
                    if existing_type is None or existing_type != col_type:
                        is_metadata_different = True
                        break
                cur_dict_basket_columns = getattr(metadata, "dict_basket_columns", None)
                if cur_dict_basket_columns or existing_meta_dict_columns:
                    if cur_dict_basket_columns is None or existing_meta_dict_columns is None:
                        is_metadata_different = True
                    else:
                        for col_name, col_info in metadata.dict_basket_columns.items():
                            if not existing_meta_dict_columns:
                                is_metadata_different = True
                                break
                            existing_meta_column_info = existing_meta_dict_columns.get(col_name)
                            if existing_meta_column_info is None:
                                is_metadata_different = True
                                break
                            existing_type = existing_meta_column_info.value_type
                            cur_type = col_info.value_type

                            if issubclass(existing_type, csp.Struct):
                                if not issubclass(cur_type, csp.Struct):
                                    is_metadata_different = True
                                    break
                                existing_meta = existing_type.metadata()
                                for field, field_type in cur_type.metadata().items():
                                    if existing_meta.get(field) != field_type:
                                        is_metadata_different = True
                            else:
                                is_metadata_different = existing_type is not cur_type

            if is_metadata_different:
                read_lock.unlock()
                raise RuntimeError(
                    f"Metadata mismatch at {metadata_file_path}\nCurrent:\n{metadata}\nExisting:{existing_meta}\n"
                )
            return read_lock

    def get_partition(self, partition_values: Dict[str, object]):
        """Get a partition object that corresponds to the given instance of partition key->value mapping.
        :param partition_values:
        """
        return ManagedDatasetPartition(self, partition_values)

    @property
    def category(self):
        return self._category

    @property
    def parent_folder(self):
        if self._data_paths is None:
            return None
        return self._data_paths.parent_folder

    def _set_folders(self, parent_folder, read_folders):
        assert self._data_paths is None
        if parent_folder:
            self._data_paths = DatasetPaths(
                parent_folder=parent_folder,
                read_folders=read_folders,
                name=self._name,
                data_category=self._category,
                time_aggregation=self.metadata.time_aggregation,
            )
        else:
            assert not read_folders, "Provided read folders without parent folder"
        self._lock_context = None

    @property
    def data_paths(self) -> DatasetPaths:
        return self._data_paths

    @property
    def metadata(self):
        return self._metadata
