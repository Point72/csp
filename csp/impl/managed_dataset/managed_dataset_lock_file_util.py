import os
import threading
import typing
from contextlib import contextmanager

from csp.utils.file_permissions import create_folder_with_permissions
from csp.utils.lock_file import LockFile

if typing.TYPE_CHECKING:
    from .managed_dataset import ManagedDataset


class LockContext:
    def __init__(self, data_set: "ManagedDataset"):
        self._data_set = data_set

    def resolve_lock_file_path_and_create_folders(self, file_path: str, use_read_folders: bool):
        parent_folder, lock_file_path = self._data_set.data_paths.resolve_lock_file_path(
            file_path, use_read_folders=use_read_folders
        )
        # We need to make sure that the root folder is created with the right permissions
        create_folder_with_permissions(parent_folder, self._data_set.cache_config.data_file_permissions)
        create_folder_with_permissions(
            os.path.dirname(lock_file_path), self._data_set.cache_config.lock_file_permissions
        )
        return lock_file_path


class ManagedDatasetLockUtil:
    _READ_WRITE_LOCK_FILE_NAME = ".csp_read_write_lock"
    _MERGE_LOCK_FILE_NAME = ".csp_merge_lock"
    _TLS = threading.local()

    def __init__(self, lock_file_permissions):
        self._lock_file_permissions = lock_file_permissions

    @classmethod
    @contextmanager
    def set_dataset_context(cls, lock_context: LockContext):
        prev = getattr(cls._TLS, "instance", None)
        try:
            cls._TLS.instance = lock_context
            yield lock_context
        finally:
            if prev is not None:
                cls._TLS.instance = prev
            else:
                delattr(cls._TLS, "instance")

    @classmethod
    def get_cur_context(cls):
        res = getattr(cls._TLS, "instance", None)
        if res is None:
            raise RuntimeError("Trying to get lock context without any context set")
        return res

    def _create_lock(self, file_path, lock_name, shared, is_lock_in_root_folder, timeout_seconds, retry_period_seconds):
        cur_context = self.get_cur_context()
        if os.path.isfile(file_path) or is_lock_in_root_folder:
            base_path = os.path.splitext(os.path.basename(file_path))[0]
            dir_name = os.path.dirname(file_path)
            lock_file_name = f"{lock_name}.{base_path}"
            return LockFile(
                file_path=cur_context.resolve_lock_file_path_and_create_folders(
                    os.path.join(dir_name, lock_file_name), use_read_folders=shared
                ),
                shared=shared,
                file_permissions=self._lock_file_permissions,
                timeout_seconds=timeout_seconds,
                retry_period_seconds=retry_period_seconds,
            )
        else:
            return LockFile(
                file_path=cur_context.resolve_lock_file_path_and_create_folders(
                    os.path.join(file_path, lock_name), use_read_folders=shared
                ),
                shared=shared,
                file_permissions=self._lock_file_permissions,
                timeout_seconds=timeout_seconds,
                retry_period_seconds=retry_period_seconds,
            )

    def write_lock(self, file_path, is_lock_in_root_folder=None, timeout_seconds=None, retry_period_seconds=None):
        return self._create_lock(
            file_path,
            lock_name=self._READ_WRITE_LOCK_FILE_NAME,
            shared=False,
            is_lock_in_root_folder=is_lock_in_root_folder,
            timeout_seconds=timeout_seconds,
            retry_period_seconds=retry_period_seconds,
        )

    def read_lock(self, file_path, is_lock_in_root_folder=None, timeout_seconds=None, retry_period_seconds=None):
        return self._create_lock(
            file_path,
            lock_name=self._READ_WRITE_LOCK_FILE_NAME,
            shared=True,
            is_lock_in_root_folder=is_lock_in_root_folder,
            timeout_seconds=timeout_seconds,
            retry_period_seconds=retry_period_seconds,
        )

    def merge_lock(self, file_path, is_lock_in_root_folder=None, timeout_seconds=None, retry_period_seconds=None):
        return self._create_lock(
            file_path,
            lock_name=self._MERGE_LOCK_FILE_NAME,
            shared=False,
            is_lock_in_root_folder=is_lock_in_root_folder,
            timeout_seconds=timeout_seconds,
            retry_period_seconds=retry_period_seconds,
        )
