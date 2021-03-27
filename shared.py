import platform
import numpy as np
from ctypes import cdll, c_long, c_int, POINTER, c_uint8, cast, c_uint16, sizeof, c_uint64, c_int32, c_int64
import multiprocessing as mp
from time import time
import os
from queue import Empty, Full

try:
    from multiprocessing.shared_memory import SharedMemory
except:
    is_linux = platform.system().lower().startswith("linux")
    assert is_linux, "Python version >= 3.8 is required"
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
        if not os.path.exists(base_name):
            os.makedirs(".shm/%s" % base_name, exist_ok=True)
        if number is None:
            number = int("".join(map(lambda x: str(ord(x)), base_name))) % 251
            number = c_uint8(number)
        base_name = os.path.abspath(base_name)
        return dll.ftok(base_name, number)


    def get_cur_key(number):
        return get_key("./", number)


    def create_shm(key: int, length: int, c_type):
        size = length * sizeof(c_type)
        shm_id = dll.shmget(key, size, IPC_CREAT | 0o666)
        assert shm_id != -1, "create shared memory failed"
        p = dll.shmat(shm_id, 0, 0)
        return p, shm_id


    def as_array(p, length, c_type, offset=0):
        """
        Note: offset is in bytes.
        注意 offset 是以字节为单位的
        """
        type_p = POINTER(c_type * length)
        p = cast(p, type_p)
        arr = np.ndarray((length,), c_type, p.contents, offset)
        return arr


    class SharedMemory:
        def __init__(self, name: str, create: bool, size: int):
            """
            Used in place of the built-in SharedMemory, since Python versions smaller than 3.8 do not have this module.
            用于代替内置的SharedMemory，因为小于3.8版本的python 没有这个模块
            @param name:
            @param create:
            @param size:
            """
            self.name = name
            self.size = size
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
            print("detach:", dll.shmdt(self._pointer))
            print("release shared memory:", dll.shmctl(self.shm_id, IPC_RMID, None))

        def __reduce__(self):
            return self.__class__, (self.name, False, self.size)

        def __del__(self):
            self.close()


    class Value(np.ndarray):
        def __init__(self, ident: int, length, c_type):
            """
            Creates an numpy array of shared memory for the ident.
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
            print("detach:", dll.shmdt(self._pointer))
            print("release shared memory:", dll.shmctl(self.shm_id, IPC_RMID, None))

        def __reduce__(self):
            return self.__class__, (self.key, self.length, self.c_type)

        def __del__(self):
            self.close()


class Pickler:
    """
    Used to convert the queue fetch result NumPy Array to the desired object and to convert the sent object to NumPy Array when sending through the queue.
    用于将队列取出结果numpy array转换为想要的对象和在通过队列发送时将发送的对象转换为numpy array
    """

    def __init__(self, item_size):
        """
        @param item_size: The size of each piece of data. 每条数据的大小  
        """
        self.item_size = item_size

    def assign(self, buffer, data) -> None:
        """
        The data to be sent PUT is assigned to the NumPy Array.
        将要发送put的数据赋值给numpy array
        @param buffer: numpy array.
        @param data: The data to be sent. 要发送的数据 
        @return:
        """
        raise NotImplemented("Not Implements")

    def loads(self, arr):
        """
        Converts the numpy array of the result of get to the desired object.
        将get的结果numpy array转换为想要的对象
        @param arr:
        @return:
        """
        raise NotImplemented("Not Implements")


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

    def alloc(self, sm_name, create):
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


class Queue:
    def __init__(self, namespace, pickler, maxsize, args=None):
        """
        This is a queue for interprocess communication.
        这是一个进程间通信的队列
        @param namespace: Give shared memory a unique name. 给共享内存起个唯一的名字 
        @param pickler: Pickler.
        @param maxsize: Maximum queue length, note that this will be directly claimed as shared memory. 队列最大长度，注意这将被直接申请为共享内存 
        @param args: None.
        """
        if args is None:
            lock = not_empty = not_full = args
        else:
            lock, not_empty, not_full = args

        item_size = pickler.item_size

        self.lock = lock or mp.RLock()
        self.item_size = item_size
        self.maxsize = maxsize
        self.namespace = namespace

        shared_structure = SharedStructure()
        shared_structure.add(name="cursor", length=6, c_type=c_int64)
        shared_structure.add(name="buffer", length=item_size * maxsize, c_type=c_uint8)
        self.sm, res = shared_structure.alloc(namespace, args is None)
        self.cursor = res.cursor
        self.buffer = res.buffer.reshape((-1, item_size))

        if args is None:
            self.cursor[:] = 0

        self.pickler: Pickler = pickler
        self.not_empty = not_empty or mp.Condition(self.lock)
        self.not_full = not_full or mp.Condition(self.lock)

    def __reduce__(self):
        return self.__class__, (self.namespace, self.pickler, self.maxsize,
                                (self.lock, self.not_empty, self.not_full))

    def __len__(self):
        return self.length

    def __del__(self):
        self.sm.close()

    @property
    def start(self):
        return self.cursor[0]

    @start.setter
    def start(self, v):
        self.cursor[0] = v

    @property
    def end(self):
        return self.cursor[2]

    @end.setter
    def end(self, v):
        self.cursor[2] = v

    @property
    def length(self):
        return self.cursor[4]

    @length.setter
    def length(self, v):
        self.cursor[4] = v

    def _qsize(self):
        return len(self)

    def qsize(self):
        return self._qsize()

    def full(self):
        return self.cursor[2] >= self.maxsize

    def empty(self):
        return self.cursor[2] <= 0

    def _put(self, data):
        if self.length >= self.maxsize:
            raise Full
        self.pickler.assign(self.buffer[self.end], data)
        self.end = (self.end + 1) % self.maxsize
        self.length += 1
        return True

    def _get(self):
        if self.length <= 0:
            raise Empty
        data = self.pickler.loads(self.buffer[self.start])
        self.start = (self.start + 1) % self.maxsize
        self.length -= 1
        return data

    def put(self, item, block=True, timeout=None):
        with self.not_full:
            if not block:
                if self._qsize() >= self.maxsize:
                    raise Full
            elif timeout is None:
                while self._qsize() >= self.maxsize:
                    self.not_full.wait()
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time() + timeout
                while self._qsize() >= self.maxsize:
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Full
                    self.not_full.wait(remaining)
            self._put(item)
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
            item = self._get()
            self.not_full.notify()
            return item

    def get_nowait(self):
        with self.lock:
            return self._get()

    def put_nowait(self, data):
        with self.lock:
            return self._put(data)
