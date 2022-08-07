import os
import time

base_dir = os.path.dirname(os.path.abspath(__file__))

try:
    from filelock import FileLock
except:
    class FileLock:
        def __init__(self, path):
            self.__path = path

        @property
        def lock_file(self):
            return self.__path

        def create_file(self):
            try:
                open(self.__path, "x")
                return True
            except:
                return False

        def delete_file(self):
            try:
                os.remove(self.__path)
                return True
            except:
                return False

        def lock(self, timeout=None):
            future = time.time() + timeout if timeout is not None else 0xffff
            while not self.create_file():
                time.sleep(0.005)
                if timeout is not None and time.time() < future:
                    return False
            return True

        def unlock(self, timeout=None):
            future = time.time() + timeout if timeout is not None else 0xffff
            while not self.delete_file():
                time.sleep(0.005)
                if timeout is not None and time.time() < future:
                    return False
            return True

        acquire = lock
        release = unlock

        def __enter__(self):
            self.lock()
            # print("__enter__")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # print("__exit__")
            self.unlock()
