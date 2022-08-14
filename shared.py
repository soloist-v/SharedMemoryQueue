import platform
import warnings
from typing import Any

import numpy as np
from ctypes import cdll, c_long, c_int, POINTER, c_uint8, cast, c_uint16, sizeof, c_uint64, c_int32, c_int64, c_uint32
from numpy import ndarray, uint8, uint32
from pickle import dump, loads
import multiprocessing as mp
import hashlib
import mmap
import os
import secrets

if os.name == "nt":
    import _winapi

    _USE_POSIX = False
else:
    # import _posixshmem

    _USE_POSIX = True
_SHM_SAFE_NAME_LENGTH = 14

# Shared memory block name prefix
if _USE_POSIX:
    _SHM_NAME_PREFIX = '/psm_'
else:
    _SHM_NAME_PREFIX = 'wnsm_'


def make_filename():
    "Create a random filename for the shared memory object."
    # number of random bytes to use for name
    nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
    assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
    name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
    assert len(name) <= _SHM_SAFE_NAME_LENGTH
    return name


def _make_filename():
    "Create a random filename for the shared memory object."
    # number of random bytes to use for name
    nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
    assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
    name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
    assert len(name) <= _SHM_SAFE_NAME_LENGTH
    return name


try:
    from multiprocessing.shared_memory import SharedMemory
except ImportError:
    is_win = platform.system().lower().startswith("win")
    if is_win:
        _SHM_SAFE_NAME_LENGTH = 14
        _SHM_NAME_PREFIX = 'wnsm_'


        def _make_filename():
            nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2  # 4
            assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
            name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
            assert len(name) <= _SHM_SAFE_NAME_LENGTH
            return name


        class SharedMemory:
            def __init__(self, name=None, create=False, size=0):
                temp_name = _make_filename() if name is None else name
                self._mmap = mmap.mmap(-1, size, tagname=temp_name)
                self._name = temp_name
                self._size = size
                self._buf = memoryview(self._mmap)

            def __del__(self):
                try:
                    self.close()
                except OSError:
                    pass

            def __reduce__(self):
                return self.__class__, (self.name, False, self.size,)

            def __repr__(self):
                return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

            @property
            def buf(self):
                return self._buf

            @property
            def name(self):
                reported_name = self._name
                return reported_name

            @property
            def size(self):
                "Size in bytes."
                return self._size

            def close(self):
                if self._buf is not None:
                    self._buf.release()
                    self._buf = None
                if self._mmap is not None:
                    self._mmap.close()
                    self._mmap = None

            def unlink(self):
                warnings.warn(
                    "The current version of Python does not implement the unlink method, "
                    "so the shared memory cannot be released. You are advised to upgrade to Python 3.8 or higher")
                self.close()

    else:
        try:
            dll = cdll.LoadLibrary("librt.so")
        except:
            dll = cdll.LoadLibrary("librt.so.1")

        IPC_CREAT = 0o0001000
        IPC_EXCL = 0o0002000
        IPC_NOWAIT = 0o0004000
        IPC_RMID = 0

        uint8_p = POINTER(c_uint8)
        uint16_p = POINTER(c_uint16)
        dll.ftok.restype = c_long
        dll.shmget.restype = c_int
        dll.shmat.restype = uint8_p
        dll.shmdt.restype = c_int
        dll.shmctl.restype = c_int


        def get_key(base_name, number=None):
            md5 = hashlib.md5()
            md5.update(base_name.encode('utf-8'))
            return int(md5.hexdigest()[::4], 16)


        def get_cur_key(number):
            return get_key("./", number)


        def create_shm(key: int, length: int, c_type):
            size = length * sizeof(c_type)
            shm_id = dll.shmget(key, size, IPC_CREAT | 0o666)
            assert shm_id != -1, "create shared memory failed. please check shared name."
            p = dll.shmat(shm_id, 0, 0)
            return p, shm_id


        def as_array(p, length, c_type, offset=0):
            """
            注意 offset 是以字节为单位的
            """
            type_p = POINTER(c_type * length)
            p = cast(p, type_p)
            arr = np.ndarray((length,), c_type, p.contents, offset)
            return arr


        class SharedMemory:
            def __init__(self, name: str, create: bool, size: int):
                """
                用于代替内置的SharedMemory，因为小于3.8版本的python 没有这个模块
                @param name:
                @param create:
                @param size:
                """
                name = _make_filename() if name is None else name
                self.name = name
                self.size = size
                if name.isdigit():
                    key = int(name)
                else:
                    key = get_key(name)
                self._pointer, self.shm_id = create_shm(key, size, c_uint8)
                type_p = POINTER(c_uint8 * size)
                self._buf = cast(self._pointer, type_p).contents

            @property
            def buf(self):
                return self._buf

            def as_array(self, shape=None):
                return np.ctypeslib.as_array(self._buf, shape)

            def close(self):
                dll.shmdt(self._pointer)
                dll.shmctl(self.shm_id, IPC_RMID, None)
                # print("detach: %s", )
                # print("release shared memory: %s", )
                # import subprocess as sp
                # sp.call(f"ipcrm -m {self.shm_id}", shell=True)

            def unlink(self):
                self.close()

            def __reduce__(self):
                return self.__class__, (self.name, False, self.size)

            def __del__(self):
                self.close()


        class Value(np.ndarray):
            def __init__(self, ident: int, length, c_type):
                """
                Creates an numpy array of shared memory for the ident
                根据ident创建一个基于共享内存的numpy数组
                @param ident:
                @param length:
                @param c_type:
                """
                self.key = ident
                self.length = length
                self.c_type = c_type
                self.size = length * sizeof(c_type)
                self._pointer, self.shm_id = create_shm(ident, length * sizeof(c_type), c_uint8)
                type_p = POINTER(c_type * length)
                self._buf = cast(self._pointer, type_p).contents
                super().__init__((length,), c_type, self._buf)

            def close(self):
                print("detach: %s", dll.shmdt(self._pointer))
                print("release shared memory: %s", dll.shmctl(self.shm_id, IPC_RMID, None))

            def __reduce__(self):
                return self.__class__, (self.key, self.length, self.c_type)

            def __del__(self):
                self.close()


class Dict(object):

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, item):
        return object.__getattribute__(self, item)

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class ArrayIO:
    def __init__(self, arr: np.ndarray):
        self.buffer = arr
        self.cursor = 0

    def read(self, n) -> Any:
        data = self.buffer[self.cursor: self.cursor + n]
        self.cursor += len(data)
        return data

    def readline(self) -> Any:
        pass

    def write(self, data) -> int:
        data = np.frombuffer(data, uint8)
        data_len = len(data)
        end = self.cursor + data_len
        self.buffer[self.cursor: end] = data
        self.cursor = end
        return data_len

    def __len__(self):
        return self.cursor


def read_data(buffer: ndarray):
    buffer = buffer.view("uint8")
    data_len = buffer[:4].view(uint32)[0]
    res_data = buffer[4: 4 + data_len]
    if not len(res_data):
        return None
    return loads(res_data)


def write_data(buffer: ndarray, obj):
    buffer = buffer.reshape(-1).view(uint8)
    file = ArrayIO(buffer[4:])
    dump(obj, file, -1)
    data_len = len(file)
    buffer[:4] = np.array((data_len,), uint32).view(dtype=uint8)
    return data_len


class ReadWriteLock:
    def __init__(self, **kwargs):
        self.read_num = kwargs.get("read_num", mp.Value(c_int32, 0))
        self.lock = kwargs.get("lock", mp.Lock())

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, *args):
        return self.lock.__exit__(*args)

    def write_acquire(self, block=True, timeout=None):
        return self.lock.acquire(block, timeout=timeout)

    def write_release(self):
        return self.lock.release()

    def read_acquire(self):
        with self.read_num.get_lock():
            self.read_num.value += 1
            if self.read_num.value == 1:
                return self.lock.acquire()

    def read_release(self):
        with self.read_num.get_lock():
            self.read_num.value -= 1
            if self.read_num.value == 0:
                return self.lock.release()

    def __getstate__(self):
        return {"lock": self.lock, "read_num": self.read_num}

    def __setstate__(self, state):
        self.__init__(**state)
