from shared import Queue


def task(queue: Queue):
    for i in range(10):
        data = queue.get()
        print("cur get:", data)


if __name__ == '__main__':
    import multiprocessing as mp
    import time

    t_que = Queue(1024 * 4)
    proc = mp.Process(target=task, args=(t_que,))
    proc.start()
    for i in range(10):
        t_que.put("number: i=%s" % i)
        time.sleep(1)
