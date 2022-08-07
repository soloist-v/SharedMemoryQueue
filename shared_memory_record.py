import json
import os
from _filelock import FileLock

base_dir = os.path.abspath(os.path.dirname(__file__))


class SharedMemoryRecorder:
    cache_names_file = f"{base_dir}/.sm_names"
    lock_file = f"{base_dir}/.lock"
    lock = FileLock(lock_file)

    @classmethod
    def load_cache(cls):
        if os.path.exists(cls.cache_names_file):
            return json.loads(open(cls.cache_names_file, 'rb').read())
        return []

    @classmethod
    def remove_lock_file(cls):
        if os.path.exists(cls.lock_file):
            os.remove(cls.lock_file)

    @classmethod
    def release_last_sm(cls):
        from shared import SharedMemory
        with cls.lock:
            if os.path.exists(cls.cache_names_file):
                _is_error = False
                for name, size, shm_id in cls.load_cache():
                    try:
                        sm = SharedMemory(name, False, size)
                        sm.close()
                        sm.unlink()
                    except Exception as e:
                        print(str(e).split()[-1], end=";")
                        _is_error = True
                if _is_error: print()
                os.remove(cls.cache_names_file)

    @classmethod
    def save_sm_name(cls, name, size, shm_id=None):
        with cls.lock:
            data = cls.load_cache()
            data.append([name, size, shm_id])
            open(cls.cache_names_file, 'wb').write(json.dumps(data).encode("utf8"))


def release_last_shm():
    return SharedMemoryRecorder.release_last_sm()


def release_sm(sm_name, size, shm_id):
    import subprocess
    from shared import SharedMemory
    from platform import system
    try:
        sm = SharedMemory(sm_name, create=False, size=size)
        sm.close()
        sm.unlink()
        if system().startswith("Linux"):
            subprocess.call(f"ipcrm -m {shm_id}", shell=True)
    except:
        pass
    return True
