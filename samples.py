from shared import Queue, Pickler
import numpy as np
from ctypes import cdll, c_long, c_int, POINTER, c_uint8, cast, c_uint16, sizeof, c_uint64, c_int32, c_int64


class DetPickler(Pickler):
    """
    这是一个将图片、标签、边框 同时传送的例子
    """

    def __init__(self):
        w, h, d = 1920, 1080, 3
        label_len = 300
        # 计算总大小
        size = 4 * sizeof(c_int32) + w * h * d + label_len * sizeof(c_uint16) + sizeof(c_uint16) * 4 * label_len
        super().__init__(size)

    def assign(self, buffer: np.ndarray, data) -> None:
        frame, labels, boxes = data
        labels = np.array(labels, np.uint16)
        boxes = np.array(boxes, np.uint16)
        h, w, d = frame.shape
        label_len = len(labels)
        args = np.array((h, w, d, label_len))

        l1 = len(args) * sizeof(c_int32)
        arr_arg = buffer[0:l1].view(np.int32)

        l2 = w * h * d + l1
        arr_frame = buffer[l1:l2].reshape((h, w, d))

        l3 = sizeof(c_uint16) * label_len + l2
        arr_label = buffer[l2:l3].view(np.uint16)

        l4 = sizeof(c_uint16) * label_len * 4 + l3
        arr_box = buffer[l3:l4].view(np.uint16).reshape((label_len, 4))

        arr_arg[:] = args
        arr_frame[:] = frame
        arr_label[:] = labels
        arr_box[:] = boxes

    def loads(self, arr):
        l1 = 4 * sizeof(c_int32)
        h, w, d, label_len = arr[:l1].view(np.int32)
        l2 = w * h * d + l1
        frame = arr[l1:l2].reshape((h, w, d))
        l3 = sizeof(c_uint16) * label_len + l2
        labels = arr[l2:l3].view(np.uint16)
        l4 = sizeof(c_uint16) * label_len * 4 + l3
        boxes = arr[l3:l4].view(np.uint16).reshape((label_len, 4))

        return frame, labels, boxes


class ImgPickler(Pickler):
    """
    这个例子实现了同时发送字符串、图片数组、帧号
    """

    def __init__(self):
        head = 4 * 5
        vfile = 260
        fn = 8
        w, h, d = 1920, 1080, 3
        size = head + vfile + fn + w * h * d
        self.head = head
        self.file = vfile
        self.fn = fn
        self.img = w * h * d
        super().__init__(size)

    def assign(self, buffer, data) -> None:
        vf = data["vfile"]
        vf = bytearray(vf, "utf8")
        fn = data["fn"]
        img = data["img"]

        vf_len = len(vf)
        h, w, d = img.shape
        img_len = h * w * d

        l0 = self.head
        head = buffer[:l0].view(np.int32)
        l1 = l0 + vf_len
        vfile = buffer[l0:l1]
        l2 = l1 + self.fn
        fno = buffer[l1: l2].view(np.int64)
        l3 = l2 + img_len
        frame = buffer[l2:l3]

        head[:] = vf_len, img_len, w, h, d
        vfile[:] = vf
        fno[0] = fn
        frame[:] = img.reshape((img_len,))

    def loads(self, buffer):
        res = {}
        l0 = self.head
        head = buffer[:l0].view(np.int32)
        vf_len, img_len, w, h, d = head
        l1 = l0 + vf_len
        res["vfile"] = bytes(buffer[l0:l1]).decode("utf8")
        l2 = l1 + self.fn
        res["fn"] = buffer[l1: l2].view(np.int64)[0]
        l3 = l2 + img_len
        res["img"] = buffer[l2:l3].reshape((h, w, d))
        return res


class FilepathPickler(Pickler):
    """
    这个例子将文件路径路径通过队列传递
    """

    def __init__(self):
        super().__init__(260)  # 规定filepath最大长度不能超过260

    def assign(self, buffer, data) -> None:
        data = bytearray(data, "utf8")
        assert len(data) <= self.item_size, "filepath too long"
        buffer[:len(data)] = data

    def loads(self, arr):
        arr = arr.copy()
        return bytes(arr).decode("utf8")


def task(queue: Queue):
    for i in range(10):
        data = queue.get()
        print("cur get:", data)


if __name__ == '__main__':
    import multiprocessing as mp
    import time

    pickler = FilepathPickler()
    t_que = Queue("my_queue00", pickler, 5)
    proc = mp.Process(target=task, args=(t_que,))
    proc.start()
    for i in range(10):
        t_que.put("number: i=%s" % i)
        time.sleep(1)
