import threading
import queue
import time

_SENTINEL = object()

def slow_compute_a(i):
    time.sleep(0.3)  # 模拟生成样本耗时
    return i * 10

def compute_b(x):
    time.sleep(1.0)  # 模拟更耗时的处理
    print("consume:", x)

class PrefetchIterator:
    def __init__(self, iterable, max_prefetch=2):
        self.iterable = iter(iterable)
        self.q = queue.Queue(maxsize=max_prefetch)
        self.worker = threading.Thread(target=self._producer, daemon=True)
        self.worker.start()

    def _producer(self):
        try:
            for item in self.iterable:
                self.q.put(item)
        finally:
            self.q.put(_SENTINEL)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.q.get()
        if item is _SENTINEL:
            raise StopIteration
        return item

def source():
    for i in range(5):
        yield slow_compute_a(i)

for x in PrefetchIterator(source(), max_prefetch=2):
    compute_b(x)
