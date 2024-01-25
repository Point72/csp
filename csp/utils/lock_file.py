import datetime
import fcntl
import os
import threading
import time
from typing import Dict, Optional

from csp.impl.struct import Struct
from csp.utils.file_permissions import FilePermissions, ensure_file_exists_with_permissions
from csp.utils.rm_utils import rm_file_or_folder


class _FileLockRecord(Struct):
    open_file: object
    ref_count: int
    shared: bool
    thread_lock: object


class _LockWrapper:
    """A simple wrapper. That ensures at __exit__ that lock is released.

    The point of this wrapper supports the following use case that raw lock doesn't:

    with _LockWrapper(lock) as lock_wrapper:
        lock_wrapper.release()
    """

    def __init__(self, lock):
        self._lock = lock
        self._locked = False

    def acquire(self):
        if self._locked:
            raise RuntimeError("Trying to lock wrapper twice")
        self._lock.acquire(blocking=True)
        self._locked = True

    def release(self):
        if self._locked:
            self._locked = False
            self._lock.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class _TimeoutMonitor:
    def __init__(self, timeout_seconds):
        self._timeout_seconds = timeout_seconds
        self._init_time = datetime.datetime.now()
        self._timed_out = None

    def timed_out(self):
        # We want the first call to always return False.
        if self._timed_out is None:
            self._timed_out = False
        elif not self._timed_out:
            self._timed_out = (datetime.datetime.now() - self._init_time).total_seconds() > self._timeout_seconds
        return self._timed_out

    def __bool__(self):
        return not self._timed_out


class LockFile:
    # We need to maintain it for reentrance of file lock
    _LOCKS_DICT_PID = None
    _LOCKS_DICT: Dict[str, _FileLockRecord] = {}
    _LOCKS_DICT_GUARD = threading.Lock()
    DEFAULT_RETRY_PERIOD_SECONDS = 1
    DEFAULT_TIMEOUT_SECONDS = 3600
    _CSP_LOCK_FILE_NAME = ".csp_lock"

    def __init__(
        self,
        file_path,
        timeout_seconds=None,
        retry_period_seconds=None,
        shared: bool = False,
        file_permissions: Optional[FilePermissions] = None,
    ):
        self._file_path = os.path.realpath(file_path)
        self._fd = None
        self._timeout_seconds = timeout_seconds if timeout_seconds is not None else self.DEFAULT_TIMEOUT_SECONDS
        self._retry_period_seconds = (
            retry_period_seconds if retry_period_seconds is not None else self.DEFAULT_RETRY_PERIOD_SECONDS
        )
        self._shared = shared
        self._file_permissions = file_permissions
        self._cur_object_lock_count = 0
        self._lock_record: Optional[_FileLockRecord] = None

    @property
    def file_path(self):
        return self._file_path

    @classmethod
    def csp_lock_for_folder(cls, folder_path, shared=True):
        return LockFile(os.path.join(folder_path, cls._CSP_LOCK_FILE_NAME), shared=shared)

    def __del__(self):
        while self._cur_object_lock_count > 0:
            self.unlock()

    @classmethod
    def _clean_and_get_locks_dict(cls):
        assert (not cls._LOCKS_DICT) or (cls._LOCKS_DICT_PID is not None)
        if cls._LOCKS_DICT_PID:
            cur_pid = os.getpid()
            if cur_pid != cls._LOCKS_DICT_PID:
                for record in cls._LOCKS_DICT.values():
                    record.open_file.close()
                cls._LOCKS_DICT.clear()
            cls._LOCKS_DICT_PID = cur_pid
        else:
            cls._LOCKS_DICT_PID = os.getpid()
        return cls._LOCKS_DICT

    def lock(self):
        if self._lock_record:
            with self._lock_record.thread_lock:
                self._cur_object_lock_count += 1
                self._lock_record.ref_count += 1
            return

        timeout_monitor = _TimeoutMonitor(self._timeout_seconds)

        with _LockWrapper(self._LOCKS_DICT_GUARD) as locks_dict_guard:
            locks_dict = self._clean_and_get_locks_dict()
            existing_lock_record = locks_dict.get(self._file_path)

            while self._lock_record is None:
                timed_out = timeout_monitor.timed_out()
                if existing_lock_record is not None and (not self._shared or not existing_lock_record.shared):
                    if timed_out:
                        if not self._shared or not existing_lock_record.shared:
                            if self._shared:
                                existing_lock_str, cur_lock_str = "exclusive", "shared"
                            else:
                                existing_lock_str, cur_lock_str = "shared", "exclusive"
                            raise BlockingIOError(
                                f"Unable to obtain {cur_lock_str} lock {self._file_path} within given time another {existing_lock_str} exists"
                            )
                elif existing_lock_record:
                    # There is an existing record that matches the shared lock
                    with existing_lock_record.thread_lock:
                        self._lock_record = existing_lock_record
                        self._lock_record.ref_count += 1
                        self._cur_object_lock_count += 1

                        return
                else:
                    lock_file = self._try_open_and_lock_file(self._file_path, self._shared, self._file_permissions)
                    if lock_file is not None:
                        self._lock_record = _FileLockRecord(
                            open_file=lock_file,
                            ref_count=1,
                            shared=self._shared,
                            thread_lock=threading.RLock(),
                        )
                        with self._lock_record.thread_lock:
                            locks_dict[self._file_path] = self._lock_record
                            self._cur_object_lock_count = 1
                            assert self._lock_record.ref_count == self._cur_object_lock_count
                            return
                    elif timed_out:
                        raise BlockingIOError(
                            f"Failed to obtain lock {self._file_path} withing given timedout period of {self._timeout_seconds} seconds"
                        )
                # we failed to obtain the lock, continue trying
                locks_dict_guard.release()
                time.sleep(self._retry_period_seconds)
                locks_dict_guard.acquire()
                existing_lock_record = locks_dict.get(self._file_path)

    def delete_file(self):
        if not self._lock_record:
            raise RuntimeError(f"Trying to delete unlocked file {self.file_path}")
        rm_file_or_folder(self._file_path, is_file=True)

    def unlock(self):
        if self._lock_record is None:
            return
        with self._lock_record.thread_lock:
            self._cur_object_lock_count -= 1
            self._lock_record.ref_count -= 1
            if self._cur_object_lock_count > 0:
                return

            if self._lock_record.ref_count > 0:
                self._lock_record = None
                return
        # It seems like we have to clean up but there might be some race condition and other thread obtained the lock
        # So we need to check again under both _LOCKS_DICT_GUARD and self._lock_record.thread_lock. We need to lock them in
        # order to avoid dead locks, that's why we can't do it in the block above
        with self._LOCKS_DICT_GUARD:
            with self._lock_record.thread_lock:
                if self._lock_record.ref_count == 0:
                    # We need to decrease ref count and make it negative for deallocated object, otherwise it's
                    # possible that 2 threads will end up here in rare race case:
                    # T1: acquire lock and release, decrease ref count to 0 but not get into this block yet
                    # T2: locate the same object which is still in dict, acquire and release, decrease ref count to 0 again
                    # Now both threads will see ref count of 0 and will want to cleanup, will let to do that only for the first
                    self._lock_record.ref_count -= 1
                    self._LOCKS_DICT.pop(self._file_path)
                    self._lock_record.open_file.close()
                    self._lock_record = None

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()

    @classmethod
    def _try_open_and_lock_file(cls, file_name, shared, file_permissions):
        folder = os.path.dirname(file_name)
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        num_retries = 10
        # Since we might be removing lock files while they're locked, it's possible that the file disappears on us. So we might need
        # to do a few retries here
        while True:
            try:
                if file_permissions is not None:
                    ensure_file_exists_with_permissions(file_name, file_permissions)

                fd = open(file_name, "a+")
                try:
                    exclusive_flag = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                    fcntl.lockf(fd, exclusive_flag | fcntl.LOCK_NB)
                    orig_fd_stat = os.fstat(fd.fileno())
                    cur_fd_stat = os.stat(file_name)

                    # The condition orig_fd_stat.st_ino != cur_fd_stat.st_ino is for nfs. On nfs removed file might not be removed
                    # it will be renamed instead. In this case we might still have locked non existent file. After we locked, we
                    # need to check that the id of the file on disk is the same is the id of the file that we locked.
                    # Note that on regular file system we also need this since file might have been erased and then recreated. We could just
                    # check orig_fd_stat.st_nlink but this wouldn't work on nfs. The current condition should work for both.
                    if orig_fd_stat.st_ino != cur_fd_stat.st_ino:
                        # The file was deleted
                        fd.close()
                        return None
                    return fd
                except BlockingIOError:
                    fd.close()
                    return None
                except Exception:
                    fd.close()
                    raise
            except FileNotFoundError:
                if num_retries >= 0:
                    num_retries -= 1
                else:
                    raise


class MultipleFilesLock:
    def __init__(self, file_paths_or_locks, shared: bool = False, remove_on_unlock=True):
        self._file_paths_or_locks = file_paths_or_locks
        self._shared = shared
        self._locks = None
        self._remove_on_unlock = remove_on_unlock

    def lock(self):
        """
        :return: True if the locks were succesfully locked, False otherwise
        """
        if self._locks:
            raise RuntimeError("Trying to lock MultipleFilesLock more than once")

        locks = []
        try:
            for f in self._file_paths_or_locks:
                if isinstance(f, LockFile):
                    lock = f
                else:
                    lock = LockFile(
                        file_path=f,
                        timeout_seconds=0,
                        retry_period_seconds=0,
                        shared=self._shared,
                    )
                lock.lock()
                locks.append(lock)
        except BlockingIOError:
            for lock in locks:
                lock.unlock()
            return False
        self._locks = locks
        return True

    def unlock(self):
        locks = self._locks
        self._locks = None
        if locks:
            for lock in locks:
                if self._remove_on_unlock:
                    lock.delete_file()
                lock.unlock()

    def __enter__(self):
        self.lock()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unlock()


class SimpleOneEventCondVar:
    def __init__(self):
        self._lock = threading.RLock()
        self._cond_var = threading.Condition(lock=self._lock)
        self._event_happened = False

    @property
    def lock(self):
        return self._lock

    def notify(self):
        with self._lock:
            assert not self._event_happened
            self._event_happened = True
            self._cond_var.notify_all()

    def wait(self):
        with self._lock:
            while not self._event_happened:
                self._cond_var.wait()
            self._event_happened = False
