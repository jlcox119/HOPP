from functools import wraps, partial
import time
import numpy as np
import concurrent.futures as cf
import threading
import multiprocessing
from optimization_problem_alt import HybridSizingProblem
# from collections import namedtuple

class OptimizerInterrupt(Exception):
    pass


class Worker(multiprocessing.Process):
    """
    Process worker to execute objective calculations
    """

    def __init__(self, task_queue, cache):
        super().__init__()
        self.task_queue = task_queue
        self.cache = cache

    def run(self):
        proc_name = self.name
        while True:
            # Get task from queue
            task = self.task_queue.get()

            if task is None:
                # Signal shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            # Execute task
            candidate, result = task()

            self.task_queue.task_done()
            self.cache[candidate] = result
        return


class Task(object):
    """
    Mock problem class to define the objective calculation
    """
    def __init__(self, candidate):
        self.candidate = candidate

    def __call__(self):
        result = objective(self.candidate)
        return self.candidate, result


class OptimizationDriver():
    """

    """
    DEFAULT_KWARGS = dict(time_limit=np.inf,
                          eval_limit=np.inf,
                          obj_limit=-np.inf,
                          n_proc=multiprocessing.cpu_count()-4,
                          log_freq=1,
                          log_file=None,
                          scaled=True)

    def __init__(self,
                 problem: HybridSizingProblem,
                 **kwargs) -> None:

        self.problem = problem
        self.parse_kwargs(kwargs)

        self.objective = self.wrap_objective()
        self.init_cache()
        self.get_candidate = self.problem.candidate_from_unit_array if self.options['scaled'] \
            else self.problem.candidate_from_array
        self.start_time = None

    def parse_kwargs(self, kwargs) -> None:
        self.options = self.DEFAULT_KWARGS.copy()

        for key, value in kwargs.items():
            if key in self.options:
                self.options[key] = value
            else:
                print(f'Ignoring unknown driver option {key}={value}')


    def init_cache(self):
        self.best_obj = None
        self.cache = dict()
        self.cache_info = {'hits': 0,
                           'misses': 0,
                           'size': 0,
                           'total_evals': 0}

    def check_interrupt(self, reason):
        if reason == 'time_limit':
            self.iter_start = time.time()
            elapsed = self.iter_start- self.start_time
            if elapsed > self.options['time_limit']:
                print(f"Driver exiting, time limit: {self.options['time_limit']} secs")
                raise OptimizerInterrupt

        elif reason == 'eval_limit':
            if self.eval_count > self.options['eval_limit']:
                print(f"Driver exiting, eval limit: {self.options['eval_limit']}")
                raise OptimizerInterrupt

        elif reason == 'obj_limit':
            if (self.best_obj is not None) and (self.best_obj <= self.options['obj_limit']):
                print(f"Driver exiting, obj limit: {self.options['obj_limit']}")
                raise OptimizerInterrupt

    def print_log_header(self):
        self.log_headers = ['Obj_Evals', 'Best_Objective', 'Eval_Time', 'Total_Time']
        self.log_widths = [len(header) + 5 for header in self.log_headers]

        print()
        print('##### HOPP Optimization Driver #####'.center(sum(self.log_widths)))
        print('Driver Options:', self.options, sep='\n\t')
        print('Optimizer Options:', self.opt_names, sep='\n\t')
        print()
        print("".join((val.rjust(width) for val, width in zip(self.log_headers, self.log_widths))))

    def print_log_line(self, reason=None):
        if reason == 'cache_hit':
            prefix = 'c '

        elif reason == 'new_best':
            prefix = '* '

        else:
            prefix = ''

        curr_time = time.time()
        log_values = [prefix + str(self.eval_count),
                      f'{self.best_obj:8g}',
                      f'{curr_time - self.iter_start:.2f} sec',
                      f'{curr_time - self.start_time:.2f} sec']
        print("".join((val.rjust(width) for val, width in zip(log_values, self.log_widths))))

    def print_log_end(self, best_candidate, best_objective):
        candidate_str = str(best_candidate)\
            .replace('(',   '(\n    ', 1)\
            .replace('), ', '),\n    ')\
            .replace('))',  ')\n  )')

        print()
        print(f'Best Objective: {best_objective:.2f}')
        print(f'Best Candidate:\n  {candidate_str}')

    # def get_from_cache(self, candidate):
    #     try:
    #         self.cache_info['total_evals'] += 1
    #         value = self.cache[candidate]
    #         self.cache_info['hits'] += 1
    #         self.print_log_line('cache_hit')
    #
    #         print('Sending np.nan to the optimizer')
    #         return np.nan # value
    #
    #     except KeyError:
    #         self.eval_count += 1
    #         self.cache_info['misses'] += 1
    #         return None
    #
    # def add_to_cache(self, candidate, value):
    #     self.cache[candidate] = value
    #     self.cache_info['size'] += 1

    def wrap_objective(self):
        obj = self.problem.evaluate_objective
        """
        Update with new parallel structre TODO
        """
        @wraps(obj)
        def wrapper(*args, name=None):
            self.check_interrupt('obj_limit')

            candidate = self.get_candidate(*args)

            try:
                # Check if result in cache
                self.lock.acquire()
                result = self.cache[candidate]
                self.lock.release()

                if result is None:
                    # In cache but not complete, poll cache
                    while (result := self.cache[candidate]) is None:
                        time.sleep(0.01)

                    with self.lock:
                        print(f'{name} Cache wait:', candidate, result)
                    return result['objective']

                else:
                    # Result available in cache, no work needed
                    with self.lock:
                        print(f'{name} Cache hit:', candidate, result)
                    return result['objective']

            except KeyError:
                # Candidate not in cache
                self.cache[candidate] = None  # indicates waiting in cache
                print(f'{name} Candidate entering task queue:', candidate)
                self.tasks.put(Task(candidate))
                self.lock.release()

                # Poll cache for available result (should be threading.Condition)
                while (result := self.cache[candidate]) is None:
                    time.sleep(0.01)

                with self.lock:
                    print(f'{name} Task return:', candidate, result)

                return result['objective']

        return wrapper

    def run(self, optimizers, opt_config):
        self.start_time = None
        self.eval_count = 0

        # Establish communication queues
        self.tasks = multiprocessing.JoinableQueue()
        self.manager = multiprocessing.Manager()
        self.cache = self.manager.dict()
        self.lock = threading.Lock()

        # Start workers
        n_opt = len(optimizers)
        num_workers = min(self.options['n_proc'], n_opt)
        print('Creating %d workers' % num_workers)
        workers = [Worker(self.tasks, self.cache)
                   for _ in range(num_workers)]

        for w in workers:
            w.start()

        # Starting threads that act like optimizers

        self.opt_names = [opt.__name__ for opt in optimizers]
        obj = [partial(self.objective, name=name) for name in self.opt_names]
        opt = [partial(opt, **opt_config) for opt in optimizers]

        opt[0](obj[0])
        # with cf.ThreadPoolExecutor(max_workers=n_opt) as executor:
        #     threads = {executor.submit(opt[i], obj[i]):name for i,name in enumerate(self.opt_names)}

            # for future in cf.as_completed(threads):
            #     name = threads[future]
            #     result = future.result()
            #     # check exceptions, do other stuff
            #     print(f'Optimizer {name} finished')

        # End worker processes
        for i in range(num_workers):
            self.tasks.put(None)

        # Wait for all of the tasks to finish
        self.tasks.join()
        for w in workers:
            w.join()

        best_candidate, best_result = min(self.cache.items(), key=lambda item: item[1]['objective'])
        self.print_log_end(best_candidate, best_result['objective'])

        return best_candidate, best_result['objective']