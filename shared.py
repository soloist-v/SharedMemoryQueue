import platform
import struct
import numpy as np
from ctypes import cdll, c_long, c_int, POINTER, c_uint8, cast, c_uint16, sizeof, c_uint64, c_int32, c_int64, c_uint32
import multiprocessing as mp
from time import time, sleep
from queue import Empty, Full
import hashlib
import mmap
import os
import errno
import secrets
import io
import pickle

if os.name == "nt":
    import _winapi

    _USE_POSIX = False
else:
    import _posixshmem

    _USE_POSIX = True
_SHM_SAFE_NAME_LENGTH = 14

# Shared memory block name prefix
if _USE_POSIX:
    _SHM_NAME_PREFIX = '/psm_'
else:
    _SHM_NAME_PREFIX = 'wnsm_'


def _make_filename():
    "Create a random filename for the shared memory object."
    # number of random bytes to use for name
    nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
    assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
    name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
    assert len(name) <= _SHM_SAFE_NAME_LENGTH
    return name


class _Pickler(pickle.Pickler):
    @classmethod
    def dumps(cls, obj, protocol=None):
        buf = io.BytesIO()
        cls(buf, protocol).dump(obj)
        return buf.getbuffer()

    loads = pickle.loads


def read_data(buffer: np.ndarray):
    buffer = buffer.view("uint8")
    # data_len = struct.unpack("I", buffer[:4])[0]
    data_len = buffer[:4].view(np.uint32)[0]
    res_data = buffer[4: 4 + data_len]
    return _Pickler.loads(res_data)


def write_data(buffer, data):
    buffer = buffer.view("uint8")
    data_b = _Pickler.dumps(data)
    buffer[:4] = np.array((len(data_b),), np.uint32).view(dtype="uint8")
    buffer[4: 4 + len(data_b)] = data_b
    return len(data_b)


try:
    from multiprocessing.shared_memory import SharedMemory
except ImportError:
    is_win = platform.system().lower().startswith("win")
    if is_win:
        import _winapi

        _O_CREX = os.O_CREAT | os.O_EXCL

        # FreeBSD (and perhaps other BSDs) limit names to 14 characters.
        _SHM_SAFE_NAME_LENGTH = 14

        # Shared memory block name prefix
        _SHM_NAME_PREFIX = 'wnsm_'


        def _make_filename():
            "Create a random filename for the shared memory object."
            # number of random bytes to use for name
            nbytes = (_SHM_SAFE_NAME_LENGTH - len(_SHM_NAME_PREFIX)) // 2
            assert nbytes >= 2, '_SHM_NAME_PREFIX too long'
            name = _SHM_NAME_PREFIX + secrets.token_hex(nbytes)
            assert len(name) <= _SHM_SAFE_NAME_LENGTH
            return name


        class SharedMemory:
            """Creates a new shared memory block or attaches to an existing
            shared memory block.

            Every shared memory block is assigned a unique name.  This enables
            one process to create a shared memory block with a particular name
            so that a different process can attach to that same shared memory
            block using that same name.

            As a resource for sharing data across processes, shared memory blocks
            may outlive the original process that created them.  When one process
            no longer needs access to a shared memory block that might still be
            needed by other processes, the close() method should be called.
            When a shared memory block is no longer needed by any process, the
            unlink() method should be called to ensure proper cleanup."""

            # Defaults; enables close() and unlink() to run without errors.
            _name = None
            _fd = -1
            _mmap = None
            _buf = None
            _flags = os.O_RDWR
            _mode = 0o600
            _prepend_leading_slash = False

            def __init__(self, name=None, create=False, size=0):
                if not size >= 0:
                    raise ValueError("'size' must be a positive integer")
                if create:
                    self._flags = _O_CREX | os.O_RDWR
                if name is None and not self._flags & os.O_EXCL:
                    raise ValueError("'name' can only be None if create=True")
                # Windows Named Shared Memory
                if create:
                    while True:
                        temp_name = _make_filename() if name is None else name
                        # Create and reserve shared memory block with this name
                        # until it can be attached to by mmap.
                        h_map = _winapi.CreateFileMapping(
                            _winapi.INVALID_HANDLE_VALUE,
                            _winapi.NULL,
                            _winapi.PAGE_READWRITE,
                            (size >> 32) & 0xFFFFFFFF,
                            size & 0xFFFFFFFF,
                            temp_name
                        )
                        try:
                            last_error_code = _winapi.GetLastError()
                            if last_error_code == _winapi.ERROR_ALREADY_EXISTS:
                                if name is not None:
                                    raise FileExistsError(
                                        errno.EEXIST,
                                        os.strerror(errno.EEXIST),
                                        name,
                                        _winapi.ERROR_ALREADY_EXISTS
                                    )
                                else:
                                    continue
                            self._mmap = mmap.mmap(-1, size, tagname=temp_name)
                        finally:
                            _winapi.CloseHandle(h_map)
                        self._name = temp_name
                        break

                else:
                    self._name = name
                    # Dynamically determine the existing named shared memory
                    # block's size which is likely a multiple of mmap.PAGESIZE.
                    h_map = _winapi.OpenFileMapping(_winapi.FILE_MAP_READ, False, name)
                    try:
                        p_buf = _winapi.MapViewOfFile(h_map, _winapi.FILE_MAP_READ, 0, 0, 0)
                    finally:
                        _winapi.CloseHandle(h_map)
                    size = _winapi.VirtualQuerySize(p_buf)
                    self._mmap = mmap.mmap(-1, size, tagname=name)

                self._size = size
                self._buf = memoryview(self._mmap)

            def __del__(self):
                try:
                    self.close()
                except OSError:
                    pass

            def __reduce__(self):
                return (
                    self.__class__, (self.name, False, self.size,),
                )

            def __repr__(self):
                return f'{self.__class__.__name__}({self.name!r}, size={self.size})'

            @property
            def buf(self):
                "A memoryview of contents of the shared memory block."
                return self._buf

            @property
            def name(self):
                "Unique name that identifies the shared memory block."
                reported_name = self._name
                return reported_name

            @property
            def size(self):
                "Size in bytes."
                return self._size

            def close(self):
                """Closes access to the shared memory from this instance but does
                not destroy the shared memory block."""
                if self._buf is not None:
                    self._buf.release()
                    self._buf = None
                if self._mmap is not None:
                    self._mmap.close()
                    self._mmap = None

            def unlink(self):
                """Requests that the underlying shared memory block be destroyed.
                In order to ensure proper cleanup of resources, unlink should be
                called once (and only once) across all processes which have access
                to the shared memory block."""
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


        def get_key(base_name):
            md5 = hashlib.md5()
            md5.update(base_name.encode('utf-8'))
            return int(md5.hexdigest()[::4], 16)


        def get_cur_key(number):
            return get_key("./")


        def create_shm(key: int, length: int, c_type):
            size = length * sizeof(c_type)
            shm_id = dll.shmget(key, size, IPC_CREAT | 0o666)
            assert shm_id != -1, "create shared memory failed. please check shared name."
            p = dll.shmat(shm_id, 0, 0)
            return p, shm_id


        def as_array(p, length, c_type, offset=0):
            """
            Note that offset is in bytes
            注意 offset 是以字节为单位的
            """
            type_p = POINTER(c_type * length)
            p = cast(p, type_p)
            arr = np.ndarray((length,), c_type, p.contents, offset)
            return arr


        class SharedMemory:
            def __init__(self, name: str, create: bool, size: int):
                """
                Used to replace the built-in SharedMemory, because Python versions less than 3.8 do not have this module
                用于代替内置的SharedMemory，因为小于3.8版本的python 没有这个模块
                @param name:
                @param create:
                @param size:
                """
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
                print("detach: %s", dll.shmdt(self._pointer))
                print("release shared memory: %s", dll.shmctl(self.shm_id, IPC_RMID, None))

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

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


class SharedStructure(object):
    def __init__(self, var_ls=()):
        self.__vars = []
        self.__total_size = 0
        for v in var_ls:
            self.add(*v)

    def add(self, name, length, c_type):
        size = length * sizeof(c_type)
        self.__total_size += size
        self.__vars.append((name, length, c_type, size))

    def alloc(self, sm_name=None, create=True):
        sm_name = sm_name or _make_filename()
        try:
            sm = SharedMemory(sm_name, create, self.__total_size)
        except:
            sm = SharedMemory(sm_name, not create, self.__total_size)
        buffer = sm.buf
        res = Dict()
        offset = 0
        for name, length, c_type, size in self.__vars:
            res[name] = np.ndarray((length,), c_type, buffer, offset=offset)
            offset += size
        return sm, res


class ReadWriteLock:
    def __init__(self, **kwargs):
        self.read_num = kwargs.get("read_num", mp.Value(c_int32, 0))
        self.lock = kwargs.get("lock", mp.Lock())

    def __enter__(self):
        return self.lock.__enter__()

    def __exit__(self, *args):
        return self.lock.__exit__(*args)

    def write_acquire(self):
        return self.lock.acquire()

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


def is_power_of_2(n):
    return n != 0 and ((n & (n - 1)) == 0)


def roundup_pow_of_two(n):
    position = 0
    x = n - 1
    if 0 != x:
        while True:
            x >>= 1
            position += 1
            if x == 0:
                break
    else:
        position = -1
        position += 1
    return 1 << position


class Queue:
    @property
    def _in(self):
        return self.__in[0]

    @_in.setter
    def _in(self, val):
        self.__in[0] = c_uint32(val).value

    @property
    def _out(self):
        return self.__out[0]

    @_out.setter
    def _out(self, val):
        self.__out[0] = val

    def __init__(self, buffer_size):
        """
        @param buffer_size: the size of shared memory
        """
        size = buffer_size
        self.init(size)

    def __getstate__(self):
        return self.size, self.sm.name, self.lock, self.not_full, self.not_empty

    def __setstate__(self, state):
        self.init(*state)

    def init(self, size, name=None, lock=None, not_full=None, not_empty=None):
        self.size = size
        if self.size & (self.size - 1):
            self.size = roundup_pow_of_two(self.size)
        self.lock = lock or mp.Lock()
        self.not_full = not_full or mp.Condition(self.lock)
        self.not_empty = not_empty or mp.Condition(self.lock)
        ss = SharedStructure()
        ss.add("in_", 1, c_uint32)
        ss.add("out_", 1, c_uint32)
        ss.add("buf", self.size, c_uint8)
        sm, d = ss.alloc(name)
        self.sm = sm
        self.__in = d.in_
        self.__out = d.out_
        self._buffer = d.buf

    def _qsize(self):
        _in = (self._in & (self.size - 1))
        _out = (self._out & (self.size - 1))
        if _in >= _out:
            return _in - _out
        else:
            return self.size - (_out - _in)

    def __put(self, data):
        length = len(data)
        length = min(length, self.size - self._in + self._out)
        l = min(length, self.size - (self._in & (self.size - 1)))
        st = (self._in & (self.size - 1))
        self._buffer[st:st + l] = data[:l]
        self._buffer[:length - l] = data[l:]
        self._in += length
        return length

    def __get(self, length):
        length = min(length, self._in - self._out)
        l = min(length, self.size - (self._out & (self.size - 1)))
        st = (self._out & (self.size - 1))
        buffer = self._buffer[st: st + l]
        buffer1 = self._buffer[: length - l]
        res = np.concatenate([buffer, buffer1])
        self._out += length
        return res

    def _put(self, data):
        data_len = np.array([len(data)], np.uint32).view(np.uint8)
        self.__put(data_len)
        self.__put(data)

    def _get(self):
        data_len = self.__get(4).view(np.uint32)[0]
        data = self.__get(data_len).tobytes()
        return _Pickler.loads(data)

    def put(self, item, block=True, timeout=None):
        data = _Pickler.dumps(item)
        with self.not_full:
            if not block:
                if self.size - self._qsize() < len(data):
                    raise Full
            elif timeout is None:
                while self.size - self._qsize() < len(data):
                    self.not_full.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while self.size - self._qsize() < len(data):
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Full
                    self.not_full.wait(remaining)
            # **********************
            self._put(data)
            # **********************
            self.not_empty.notify()

    def get(self, block=True, timeout=None):
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise Empty
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while not self._qsize():
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            # **********************
            data_obj = self._get()
            # **********************
            self.not_full.notify()
            return data_obj

    def get_nowait(self):
        self.get(False)

    def put_nowait(self, data):
        self.put(data, False)
