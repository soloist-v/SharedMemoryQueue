from SharedArray import zeros
from numpy import uint8
from time import sleep
from shared import read_data1, write_data1


class LockFreeQueue:
    def __init__(self, item_size, m_size=2):
        assert m_size >= 2, "The queue size must be >= 2"
        self.m_size = m_size
        self.idx_head = zeros(1, int)
        self.idx_tail = zeros(1, int)
        self.m_data = zeros((self.m_size, item_size + 4096), uint8)

    def is_empty(self):
        return self.idx_head[0] == self.idx_tail[0]

    def is_full(self):
        return self.idx_head[0] == (self.idx_tail[0] + 1) % self.m_size

    def push(self, val):
        if self.is_full():
            return False
        buffer = self.m_data[self.idx_tail[0]]
        write_data1(buffer, val)
        self.idx_tail[0] = (self.idx_tail[0] + 1) % self.m_size
        return True

    def pop(self):
        if self.is_empty():
            return False, None
        buffer = self.m_data[self.idx_head[0]]
        data = read_data1(buffer)
        self.idx_head[0] = (self.idx_head[0] + 1) % self.m_size
        return True, data

    def get(self):
        while True:
            ok, val = self.pop()
            if ok:
                return val
            sleep(0.)

    def put(self, val):
        while not self.push(val):
            sleep(0.)
