import threading
from datetime import datetime
from typing import Dict, Tuple

from csp.adapters.output_adapters.parquet import ParquetOutputConfig


class SinglePartitionFiles:
    def __init__(self, dataset_partition, parquet_output_config):
        self._dataset_partition = dataset_partition
        self._parquet_output_config = parquet_output_config
        # A mapping of (start, end)->file_path
        self._files_by_period: Dict[Tuple[datetime, datetime], str] = {}

    @property
    def dataset_partition(self):
        return self._dataset_partition

    @property
    def parquet_output_config(self):
        return self._parquet_output_config

    @property
    def files_by_period(self):
        return self._files_by_period

    def add_file(self, start_time: datetime, end_time: datetime, file_path: str):
        self._files_by_period[(start_time, end_time)] = file_path


class PartitionFileContainer:
    TLS = threading.local()

    def __init__(self, cache_config):
        # A mapping of id(dataset_partition)->(start_time,end_time)->file_path
        self._files_by_partition_and_period: Dict[int, SinglePartitionFiles] = {}
        self._cache_config = cache_config

    @classmethod
    def get_instance(cls):
        return cls.TLS.instance

    def __enter__(self):
        assert not hasattr(self.TLS, "instance")
        self.TLS.instance = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            # We don't want to finalize cache if there is an exception
            if exc_val is None:
                # First let's publish all files
                for partition_files in self._files_by_partition_and_period.values():
                    for (start_time, end_time), file_path in partition_files.files_by_period.items():
                        partition_files.dataset_partition.publish_file(
                            file_path,
                            start_time,
                            end_time,
                            self._cache_config.data_file_permissions,
                            lock_file_permissions=self._cache_config.lock_file_permissions,
                        )

                # Let's now merge whatever we can
                if self._cache_config.merge_existing_files:
                    for partition_files in self._files_by_partition_and_period.values():
                        for (start_time, end_time), file_path in partition_files.files_by_period.items():
                            partition_files.dataset_partition.merge_files(
                                start_time,
                                end_time,
                                cache_config=self._cache_config,
                                parquet_output_config=partition_files.parquet_output_config,
                            )
                            partition_files.dataset_partition.cleanup_unneeded_files(
                                start_time=start_time, end_time=end_time, cache_config=self._cache_config
                            )
        finally:
            del self.TLS.instance

    @property
    def files_by_partition(self):
        return self._files_by_partition_and_period

    def add_generated_file(
        self,
        dataset_partition,
        start_time: datetime,
        end_time: datetime,
        file_path: str,
        parquet_output_config: ParquetOutputConfig,
    ):
        key = id(dataset_partition)

        single_partition_files = self._files_by_partition_and_period.get(key)
        if single_partition_files is None:
            single_partition_files = SinglePartitionFiles(dataset_partition, parquet_output_config)
            self._files_by_partition_and_period[key] = single_partition_files
        else:
            assert single_partition_files.parquet_output_config == parquet_output_config
        single_partition_files.add_file(start_time, end_time, file_path)
