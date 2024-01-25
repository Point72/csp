import os
import tempfile
import threading
import time
import unittest

from csp.utils.lock_file import LockFile


class _SimpleOneEventCondVar:
    def __init__(self):
        self._lock = threading.Lock()
        self._cond_var = threading.Condition(lock=self._lock)
        self._event_idx = 0

    @property
    def lock(self):
        return self._lock

    def notify(self, lock=True):
        if lock:
            self._lock.acquire()
        try:
            self._event_idx += 1
            self._cond_var.notify_all()
            return self._event_idx
        finally:
            if lock:
                self._lock.release()

    def wait(self, event_value, lock=True):
        if lock:
            self._lock.acquire()
        try:
            while self._event_idx != event_value:
                self._cond_var.wait()
        finally:
            if lock:
                self._lock.release()


class TestLockFile(unittest.TestCase):
    def test_multiple_threads(self):
        for shared in [True, False]:
            for lock_top_level in [True, False]:
                fd, f_path = tempfile.mkstemp()
                os.close(fd)

                def lock_multiple(cond_var: _SimpleOneEventCondVar):
                    # Wait until the parent obtained the lock
                    cond_var.wait(1)
                    file_lock = LockFile(f_path, timeout_seconds=0, shared=shared)
                    if lock_top_level:
                        # Notify the parent that the thread started
                        cond_var.notify()
                        file_lock.lock()
                    with file_lock:
                        if not lock_top_level:
                            # Notify the parent that the thread started
                            cond_var.notify()

                        with file_lock:
                            if shared:
                                with LockFile(f_path, timeout_seconds=0, shared=shared):
                                    pass

                        if not lock_top_level:
                            cond_var.wait(3)

                    if lock_top_level:
                        cond_var.wait(3)
                        file_lock.unlock()

                cond_var = _SimpleOneEventCondVar()
                t = threading.Thread(target=lock_multiple, kwargs={"cond_var": cond_var})

                t.start()

                with cond_var.lock:
                    # Notify the thread parent has the lock
                    cond_var.notify(lock=False)

                    # Wait until we are certain that the file is locked
                    cond_var.wait(2, lock=False)

                if shared:
                    with LockFile(f_path, timeout_seconds=0, shared=shared):
                        pass
                else:
                    with self.assertRaises(BlockingIOError):
                        with LockFile(f_path, timeout_seconds=0, shared=shared):
                            pass

                cond_var.notify()
                # This can fail on a really busy machine but we want to make sure that timeout works
                try:
                    with LockFile(f_path, timeout_seconds=5, retry_period_seconds=0.01, shared=shared):
                        pass
                except BlockingIOError:
                    # VERY rare, should happen only if machine is busy, give another 30 seconds
                    with LockFile(f_path, timeout_seconds=30, retry_period_seconds=0.1, shared=shared):
                        pass

                t.join()

    def test_fork(self):
        fd, f_path = tempfile.mkstemp()
        os.close(fd)
        for shared in [True, False]:

            def fork_and_lock(lock_parent=True):
                pid = os.fork()
                if pid == 0:
                    try:
                        time.sleep(0.05)
                        with LockFile(f_path, timeout_seconds=0, shared=shared):
                            pass
                        os._exit(0)
                    except BaseException:
                        os._exit(1)
                else:
                    if lock_parent:
                        with LockFile(f_path, timeout_seconds=0, shared=shared):
                            _, res = os.waitpid(pid, 0)
                            self.assertTrue((res == 0) == shared)
                    else:
                        _, res = os.waitpid(pid, 0)
                        self.assertTrue((res == 0) == shared)

            fork_and_lock()
            with LockFile(f_path, timeout_seconds=0, shared=shared):
                fork_and_lock(lock_parent=False)


if __name__ == "__main__":
    unittest.main()
