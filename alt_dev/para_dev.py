import concurrent.futures as cf
import threading
import multiprocessing
import time
import numpy as np


class Worker(multiprocessing.Process):

    def __init__(self, task_queue, cache):
        super().__init__()
        self.task_queue = task_queue
        self.cache = cache

    def run(self):
        proc_name = self.name
        while True:
            task = self.task_queue.get()
            if task is None:
                # Poison pill means shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            candidate, result = task()

            self.task_queue.task_done()
            self.cache[candidate] = result
        return


class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def __call__(self):
        try:

            time.sleep(0.1) # pretend to take some time to do the work
            if (self.a == 9) and (self.b == 9):
                raise Exception

            return (self.a, self.b), '%s * %s = %s' % (self.a, self.b, self.a * self.b)

        except:
            return (self.a, self.b), np.nan

    def __str__(self):
        return '%s * %s' % (self.a, self.b)


def func(x, tasks, cache, lock):
    results = []

    for val in range(x):
        candidate = (val, val)

        try:
            lock.acquire()
            result = cache[candidate]
            lock.release()

            if result is None:
                while (result := cache[candidate]) is None:
                    time.sleep(0.01)

                with lock:
                    print(f'{x} Cache wait:', candidate, result)
                results.append((candidate, result))

            else:

                with lock:
                    print(f'{x} Cache hit:', candidate, result)
                results.append((candidate, result))

        except KeyError:
            cache[candidate] = None
            print(f'{x} Candidate entering task queue:', candidate)
            tasks.put(Task(*candidate))
            lock.release()

            while (result := cache[candidate]) is None:
                time.sleep(0.01)

            with lock:
                print(f'{x} Task return:', candidate, result)

            results.append((candidate, result))

    return results


def kill_workers(tasks, num_workers):
    # Add a poison pill for each consumer
    for i in range(num_workers):
        tasks.put(None)

if __name__ == '__main__':
    # Establish communication queues
    tasks = multiprocessing.JoinableQueue()
    manager = multiprocessing.Manager()
    cache = manager.dict()

    # Start workers
    num_workers = 2
    print('Creating %d workers' % num_workers)
    workers = [Worker(tasks, cache) for i in range(num_workers)]

    for w in workers:
        w.start()

    # Starting threads that act like optimizers
    start = time.perf_counter()
    lock = threading.Lock()

    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        threads = {executor.submit(func, x, tasks, cache, lock): x for x in range(10, 0, -1)}

        for future in cf.as_completed(threads):
            wait = threads[future]
            result = future.result()

            with lock:
                print(wait, 'thread finished', result)

    end = time.perf_counter()
    print(f'\n#### Elapsed time: {end - start:.2f} secs #### \n')

    # End worker processes
    kill_workers(tasks, num_workers)

    # Wait for all of the tasks to finish
    tasks.join()
    for w in workers:
        w.join()

    # Start printing results
    for key, value in cache.items():
        print('Result:', key, value)