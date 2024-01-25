from typing import Dict, List

import csp
from csp.impl.managed_dataset.managed_dataset import ManagedDataset, ManagedDatasetPartition
from csp.impl.wiring.cache_support.partition_files_container import PartitionFileContainer


class _DatasetRecord(csp.Struct):
    dataset: ManagedDataset
    read: bool = False
    write: bool = False


class RuntimeCacheManager:
    def __init__(self, cache_config, cache_data):
        self._cache_config = cache_config
        self._partition_file_container = PartitionFileContainer(cache_config)
        self._datasets: Dict[int, _DatasetRecord] = {}
        self._dataset_write_partitions: List[ManagedDatasetPartition] = []
        self._dataset_read_partitions: List[ManagedDatasetPartition] = []
        self._read_locks = []
        for graph_cache_manager in cache_data.cache_managers.values():
            if graph_cache_manager.outputs is not None:
                self.add_read_partition(graph_cache_manager.dataset_partition)
            else:
                self.add_write_partition(graph_cache_manager.dataset_partition)

    def _validate_and_lock_datasets(self):
        res = []
        for dataset_record in self._datasets.values():
            res.append(
                dataset_record.dataset.validate_and_lock_metadata(
                    lock_file_permissions=self._cache_config.lock_file_permissions,
                    data_file_permissions=self._cache_config.data_file_permissions,
                    read=dataset_record.read,
                    write=dataset_record.write,
                )
            )
        return res

    def __enter__(self):
        self._read_locks = []
        self._read_locks.extend(self._validate_and_lock_datasets())
        for partition in self._dataset_write_partitions:
            partition.create_root_folder(self._cache_config)

        self._partition_file_container.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            for lock in self._read_locks:
                lock.unlock()
        finally:
            self._read_locks.clear()

        self._partition_file_container.__exit__(exc_type, exc_val, exc_tb)

    def _add_dataset(self, dataset: ManagedDataset, read=False, write=False):
        dataset_id = id(dataset)
        dataset_record = self._datasets.get(dataset_id)
        if dataset_record is None:
            dataset_record = _DatasetRecord(dataset=dataset, read=read, write=write)
            self._datasets[dataset_id] = dataset_record
            return
        dataset_record.read |= read
        dataset_record.write |= write

    def add_write_partition(self, dataset_partition: ManagedDatasetPartition):
        self._add_dataset(dataset_partition.dataset, write=True)
        self._dataset_write_partitions.append(dataset_partition)

    def add_read_partition(self, dataset_partition: ManagedDatasetPartition):
        self._add_dataset(dataset_partition.dataset, read=True)
        self._dataset_read_partitions.append(dataset_partition)
