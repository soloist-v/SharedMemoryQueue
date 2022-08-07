from kfifo_queue import Queue
from SharedArray import SharedStructure, SharedField


def task(queue: Queue):
    for i in range(10):
        data = queue.get()
        print("cur get:", data)


def task1(a):
    print(a.a)


class Test(SharedStructure):
    def __init__(self):
        super().__init__("test_")
        self.a = SharedField(8)
        self.a.value = 12


if __name__ == '__main__':
    from multiprocessing import Process

    a = Test()
    print(a.a)
    proc = Process(target=task1, args=(a,))
    proc.start()
    proc.join()

if __name__ == '__main__':
    import multiprocessing as mp
    import time

    t_que = Queue(buffer_size=1024 * 4)
    proc = mp.Process(target=task, args=(t_que,))
    proc.start()
    for i in range(10):
        t_que.put("number: i=%s" % i)
        time.sleep(1)
