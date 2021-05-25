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
        time.sleep(0.1) # pretend to take some time to do the work
        return (self.a, self.b), '%s * %s = %s' % (self.a, self.b, self.a * self.b)

    def __str__(self):
        return '%s * %s' % (self.a, self.b)

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

    # Enqueue jobs
    num_jobs = 10
    for i in range(num_jobs):
        candidate = (1, 1)
        if candidate in cache:
            if cache[candidate] is None:
                print('Candidate waiting in queue:', candidate)

            while (result := cache[candidate]) is None:
                time.sleep(0.01)

            print('Cache hit:', candidate, result)

        else:
            print('Candidate entering task queue:', candidate)
            cache[candidate] = None
            tasks.put(Task(*candidate))

    # Add a poison pill for each consumer
    for i in range(num_workers):
        tasks.put(None)

    # Wait for all of the tasks to finish
    tasks.join()
    for w in workers:
        w.join()

    # Start printing results
    for key,value in cache.items():
        print('Result:', key, value)