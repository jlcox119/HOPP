from functools import wraps, partial
import time
import numpy as np
import concurrent.futures as cf
import threading
import multiprocessing
from optimization_problem_alt import HybridSizingProblem
import pickle


class OptimizerInterrupt(Exception):
    pass


class Worker(multiprocessing.Process):
    """
    Process worker to execute objective calculations
    """

    def __init__(self, task_queue, cache, setup):
        super().__init__()
        self.task_queue = task_queue
        self.cache = cache
        self.setup = setup

    def run(self):
        problem = self.setup()

        proc_name = self.name
        while True:
            # Get task from queue
            candidate = self.task_queue.get()

            if candidate is None:
                # Signal shutdown
                print('%s: Exiting' % proc_name)
                self.task_queue.task_done()
                break

            # Execute task
            candidate, result = problem.evaluate_objective(candidate)

            self.task_queue.task_done()
            self.cache[candidate] = result
        return


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
                 setup,
                 **kwargs) -> None:

        self.problem = setup()
        self.setup = setup
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

    def check_interrupt(self):
        self.iter_start = time.time()
        elapsed = self.iter_start- self.start_time
        if elapsed > self.options['time_limit']:
            print(f"Driver exiting, time limit: {self.options['time_limit']} secs")
            raise OptimizerInterrupt

        if self.eval_count > self.options['eval_limit']:
            print(f"Driver exiting, eval limit: {self.options['eval_limit']}")
            raise OptimizerInterrupt

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
                      # f'{curr_time - self.iter_start:.2f} sec',
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

    def write_cache(self, filename=None):
        if filename is None:
            filename = 'driver_cache.pkl'

        cache = self.cache.copy()
        with open(filename, 'wb') as f:
            pickle.dump(cache, f)

    def read_cache(self, filename=None):
        if filename is None:
            filename = 'driver_cache.pkl'

        with open(filename, 'rb') as f:
            cache = pickle.load(f)

        self.cache.update(cache)

    def wrap_objective(self):
        # obj = self.problem.evaluate_objective
        """
        Update with new parallel structre TODO
        """
        @wraps(self.wrap_objective)
        def wrapper(*args, name=None):
            self.check_interrupt()

            candidate = self.get_candidate(*args)
            self.cache_info['total_evals'] += 1

            try:
                # Check if result in cache
                self.lock.acquire()
                result = self.cache[candidate]
                self.lock.release()
                self.cache_info['hits'] += 1

                if result is None:
                    # In cache but not complete, poll cache
                    while (result := self.cache[candidate]) is None:
                        time.sleep(0.01)

                    # with self.lock:
                    #     print(f'{name} Cache wait:', candidate, result)
                    return result['objective']

                else:
                    # Result available in cache, no work needed
                    # with self.lock:
                    #     print(f'{name} Cache hit:', candidate, result)
                    return result['objective']

            except KeyError:
                # Candidate not in cache
                self.cache[candidate] = None  # indicates waiting in cache
                # print(f'{name} Candidate entering task queue:', candidate)
                self.tasks.put(candidate)
                self.lock.release()

                self.cache_info['misses'] += 1

                # Poll cache for available result (should be threading.Condition)
                while (result := self.cache[candidate]) is None:
                    time.sleep(0.01)

                if (self.best_obj is None) or (result['objective'] < self.best_obj):
                    self.best_obj = result['objective']
                    reason = 'new_best'

                else:
                    reason = ''

                with self.lock:
                    self.eval_count += 1
                    self.print_log_line(reason)
                    # print(f'{name} Task return:', candidate, result)

                self.cache_info['size'] += 1

                return result['objective']

        return wrapper

    def run(self, optimizers, opt_config, cache_file=None):
        self.start_time = time.time()
        self.eval_count = 0

        # Establish communication queues
        self.tasks = multiprocessing.JoinableQueue()
        self.manager = multiprocessing.Manager()
        self.cache = self.manager.dict()
        self.lock = threading.Lock()

        if cache_file is not None:
            self.read_cache(cache_file)

        # Start workers
        n_opt = len(optimizers)
        num_workers = min(self.options['n_proc'], n_opt)
        print('Creating %d workers' % num_workers)
        workers = [Worker(self.tasks, self.cache, self.setup)
                   for _ in range(num_workers)]

        for w in workers:
            w.start()

        # Starting threads that act like optimizers

        self.opt_names = [opt.__name__ for opt in optimizers]
        obj = [partial(self.objective, name=name) for name in self.opt_names]
        opt = [partial(opt, **opt_config) for opt in optimizers]
        for i in range(n_opt):
            obj[i].__name__ = self.opt_names[i]

        self.print_log_header()

        # opt[0](obj[0])
        with cf.ThreadPoolExecutor(max_workers=n_opt) as executor:
            threads = {executor.submit(opt[i], obj[i]):name for i,name in enumerate(self.opt_names)}

            for future in cf.as_completed(threads):
                name = threads[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (name, exc))
                else:
                    print(f'Optimizer {name} finished', data)

        # End worker processes
        for i in range(num_workers):
            self.tasks.put(None)

        # Wait for all of the tasks to finish
        self.tasks.join()
        for w in workers:
            w.join()

        best_candidate, best_result = min(self.cache.items(), key=lambda item: item[1]['objective'])
        self.print_log_end(best_candidate, best_result['objective'])
        print(self.cache)

        return best_candidate, best_result['objective']