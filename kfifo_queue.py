from SharedArray import SharedStructure, SharedField, SharedFieldUint8, SharedFieldInt32
import multiprocessing as mp
from time import time
from queue import Empty, Full
import numpy as np
from ctypes import c_uint8, c_uint32
import pickle


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
        return self.sm.in_.value

    @_in.setter
    def _in(self, val):
        self.sm.in_.value = val

    @property
    def _out(self):
        return self.sm.out_.value

    @_out.setter
    def _out(self, val):
        self.sm.out_.value = val

    class SMField(SharedStructure):
        def __init__(self, size):
            super().__init__()
            self.in_ = SharedField(1, c_uint32)
            self.out_ = SharedField(1, c_uint32)
            self.buf = SharedField(size)

    def __init__(self, buffer_size, dumps=pickle.dumps, loads=pickle.loads):
        """
        @param buffer_size: the size of shared memory
        """
        self.size = buffer_size
        if self.size & (self.size - 1):
            self.size = roundup_pow_of_two(self.size)
        self.lock = mp.Lock()
        self.not_full = mp.Condition(self.lock)
        self.not_empty = mp.Condition(self.lock)
        self.sm = self.SMField(self.size)
        self.loads = loads
        self.dumps = dumps

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
        self.sm.buf[st:st + l] = data[:l]
        self.sm.buf[:length - l] = data[l:]
        self._in += length
        return length

    def __get(self, length):
        length = min(length, self._in - self._out)
        l = min(length, self.size - (self._out & (self.size - 1)))
        st = (self._out & (self.size - 1))
        buffer = self.sm.buf[st: st + l]
        buffer1 = self.sm.buf[: length - l]
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
        return self.loads(data)

    def put(self, item, block=True, timeout=None):
        data = np.frombuffer(self.dumps(item), np.uint8)
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
        return self.get(False)

    def put_nowait(self, data):
        self.put(data, False)
