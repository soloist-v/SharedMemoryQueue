# SharedMemoryQueue

    A queue implemented using shared memory. 
    基于共享内存实现的队列
# Compare with multiprocessing Queue

    1.Windows: SharedMemoryQueue > multiprocessing.Queue > socket
    2.Linux: SharedMemoryQueue > socket > multiprocessing.Queue
# References

    1.linux kernel KFifo https://www.programmersought.com/article/4031516754/
    2.python queue.Queue
    3.python multiprocessing.shared