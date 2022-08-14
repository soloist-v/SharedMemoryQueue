from typing import List, Any, Tuple
import numpy as np
from ctypes import c_uint8, sizeof
from typing import Union
import ctypes as ct
from numpy import ndarray, prod
from shared import SharedMemory, read_data, write_data
from shared_memory_record import SharedMemoryRecorder


class _Nop:
    def __init__(self, *args, **kwargs):
        pass


class NDArray(ndarray):
    def set_data(self, data):
        write_data(self, data)

    def get_data(self):
        return read_data(self)

    @property
    def value(self):
        return self[0]

    @value.setter
    def value(self, val):
        self[0] = val


class Array(NDArray, SharedMemoryRecorder):
    """
    无论是windows还是linux共享内存的初始值均为0，所以无需后续的初始化值
    """

    def __new__(cls, shape, dtype: Union[str, np.dtype, object] = None, name=None, create=True, offset=0,
                strides=None, order=None):
        if not isinstance(dtype, np.dtype):
            dtype = np.dtype(dtype)
        size = int(np.prod(shape) * dtype.itemsize)
        buf = SharedMemory(name, create=create, size=size)
        obj = super().__new__(cls, shape, dtype, buf.buf, offset, strides, order)
        obj.buf = buf
        if create:
            cls.save_sm_name(buf.name, buf.size)
        return obj

    @property
    def name(self):
        if hasattr(self, "buf"):
            return self.buf.name
        raise Exception("此数组发生了转移/copy")

    @name.setter
    def name(self, val):
        raise Exception("Unsupported set name")

    def close(self):
        if not hasattr(self, "buf"):
            raise Exception("此数组发生了转移/copy")
        self.buf.close()
        self.buf.unlink()

    def release(self):
        self.close()

    def __reduce__(self):
        return Array, (self.shape, self.dtype, self.name, False)

    def copy(self, order='C'):
        # return np.ndarray(self.shape, self.dtype, self.data).copy(order)
        return np.copy(self.data, order)


class Value:
    def __init__(self, c_type, value=0, name=None, create=None):
        self.data = Array(1, c_type, name, create)
        self.data[0] = value

    @property
    def value(self):
        return self.data[0]


def zeros_like(a: np.ndarray, dtype=None, name=None, create=True):
    dtype = dtype or a.dtype
    shape = a.shape
    return Array(shape, dtype, name, create)


def zeros(shape, dtype=None, name=None, create=True):
    dtype = dtype or np.uint8
    return Array(shape, dtype, name, create)


def full_like(a, fill_value, dtype=None, name=None, create=True):
    dtype = dtype or a.dtype
    shape = a.shape
    arr = Array(shape, dtype, name, create)
    arr[:] = fill_value
    return arr


def full(shape, fill_value, dtype=None, name=None, create=True):
    dtype = dtype or np.uint8
    arr = Array(shape, dtype, name, create)
    arr[:] = fill_value
    return arr


def ones(shape, dtype=None, name=None, create=True):
    return full(shape, 1, dtype, name, create)


def ones_like(arr, dtype=None, name=None, create=True):
    return full_like(arr, 1, dtype, name, create)


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


class FieldMeta(type):
    def __new__(mcs, what: str, bases, attr_dict):
        bases = (*filter(lambda t: t not in (ndarray, Array), bases),)
        cls = super().__new__(mcs, what, bases, attr_dict)
        return cls


class SharedField(NDArray, metaclass=FieldMeta):
    def __init__(self, shape: Union[List, Tuple, int], c_type: Any = c_uint8, value=0):
        object.__init__(self)
        if c_type is int:
            c_type = ct.c_int
        if c_type is float:
            c_type = ct.c_float
        setattr(self, "#name", None)
        setattr(self, "#shape", shape)
        setattr(self, "#c_type", c_type)
        setattr(self, "#value", value)


class SharedFieldUint8(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, c_uint8, value)


class SharedFieldInt(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, ct.c_int, value)


class SharedFieldInt64(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, ct.c_int64, value)


class SharedFieldInt32(SharedField):
    def __init__(self, shape, value=0):
        super().__init__(shape, ct.c_int32, value)


def create_shared(self, name, create, fields: List[SharedField]):
    total_size = 0
    for field in fields:
        shape = getattr(field, "#shape")
        c_type = getattr(field, "#c_type")
        total_size += prod(shape) * sizeof(c_type)
    buf = zeros(total_size, c_uint8, name, create)
    setattr(self, "$buf", buf)
    setattr(self, "$name", buf.name)
    setattr(self, "$fields", fields)
    buffer = buf.buf.buf
    offset = 0
    for field in fields:
        name = getattr(field, "#name")
        shape = getattr(field, "#shape")
        c_type = getattr(field, "#c_type")
        value = getattr(field, "#value")
        arr = NDArray(shape, c_type, buffer, offset=offset)
        arr[:] = value
        setattr(self, name, arr)
        offset += prod(shape) * sizeof(c_type)
    return self


class SharedStructureMeta(type):
    def __call__(cls, *args, **kwargs):
        self = super().__call__(*args, **kwargs)
        name = getattr(self, "$name")
        create = getattr(self, "$create")
        fields: List[SharedField] = []
        for k, v in vars(self).items():
            if isinstance(v, SharedField):
                if k in {"get_sm_name", "close"}:
                    raise Exception("Field name error")
                # v._name = k
                setattr(v, "#name", k)
                fields.append(v)
        create_shared(self, name, create, fields)
        return self


class SharedStructure(metaclass=SharedStructureMeta):

    def __init__(self, name=None, create=True):
        setattr(self, "$name", name)
        setattr(self, "$create", create)

    def get_sm_name(self):
        return getattr(self, "$buf").name

    def close(self):
        return getattr(self, "$buf").close()

    def __getstate__(self):
        return self.get_sm_name(), getattr(self, "$fields")

    def __setstate__(self, state):
        name, fields = state
        SharedStructure.__init__(self, name, False)
        create_shared(self, name, False, fields)
