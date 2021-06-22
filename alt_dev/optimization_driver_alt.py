import time
import pickle
import queue
import numpy as np
from functools import wraps, partial

import concurrent.futures as cf
import threading
import multiprocessing


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
        # Create a new problem for the worker
        problem = self.setup()
        proc_name = self.name
        candidate = None

        while True:
            try:
                # Get task from queue
                candidate, name = self.task_queue.get()

                if candidate is None:
                    # Signal shutdown
                    print(f"{proc_name}: Exiting")
                    self.task_queue.task_done()
                    break

                # Execute task
                start_time = time.time()
                candidate, result = problem.evaluate_objective(candidate)
                result['eval_time'] = time.time() - start_time
                result['caller'] = [name]

            except KeyboardInterrupt:
                # Exit cleanly
                self.task_queue.task_done()

                # Signals any waiting optimizer threads to exit
                if candidate is not None:
                    self.cache[candidate] = OptimizerInterrupt

                print(f"{proc_name}: KeyBoardInterrupt")
                break

            # Mark task as done and return result
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
                print(f"Ignoring unknown driver option {key}={value}")


    def init_cache(self):
        self.best_obj = None
        self.cache = dict()
        self.cache_info = {'hits': 0,
                           'misses': 0,
                           'size': 0,
                           'total_evals': 0}

    def init_parallel_workers(self, num_workers):
        self.tasks = multiprocessing.JoinableQueue()
        self.manager = multiprocessing.Manager()
        self.cache = self.manager.dict()
        self.lock = threading.Lock()

        print(f"Creating {num_workers} workers")
        self.workers = [Worker(self.tasks, self.cache, self.setup)
                           for _ in range(num_workers)]

        for w in self.workers:
            w.start()

    def cleanup_parallel(self):
        if self.force_stop:
            try:
                # Mark all tasks complete
                while True:
                    self.tasks.get(block=False)
                    self.tasks.task_done()

            # Occurs when task queue is empty
            except queue.Empty:
                pass

        else:
            # Exit normally, None task signals workers to exit
            for i in range(len(self.workers)):
                self.tasks.put((None, 'worker exit'))

        # Wait for all of the tasks to finish
        self.tasks.join()
        for w in self.workers:
            w.join()

        # Any tasks returning exceptions should be removed from the cache
        pop_list = []
        for key, value in self.cache.items():
            if not isinstance(value, dict):
                pop_list.append(key)

        _ = [self.cache.pop(key) for key in pop_list]

    def check_interrupt(self):
        if self.force_stop:
            print("Driver exiting, KeyBoardInterrupt")
            raise OptimizerInterrupt

        elapsed = time.time() - self.start_time
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
        print("##### HOPP Optimization Driver #####".center(sum(self.log_widths)))
        print("Driver Options:", self.options, sep="\n\t")
        print("Optimizer Options:", self.opt_names, sep="\n\t")
        print()
        print("".join((val.rjust(width) for val, width in zip(self.log_headers, self.log_widths))))

    def print_log_line(self, info:dict):
        prefix_reasons = {'cache_hit': 'c ', 'new_best' :'* ', '': ''}
        prefix = prefix_reasons[info['reason']]


        curr_time = time.time()
        log_values = [prefix + str(self.eval_count),
                      f"{self.best_obj:8g}",
                      f"{info['eval_time']:.2f} sec",
                      f"{curr_time - self.start_time:.2f} sec"]
        print("".join((val.rjust(width) for val, width in zip(log_values, self.log_widths))))

    def print_log_end(self, best_candidate, best_objective):
        candidate_str = str(best_candidate)\
            .replace('(',   '(\n    ', 1)\
            .replace('), ', '),\n    ')\
            .replace('))',  ')\n  )')

        print()
        print(f"Best Objective: {best_objective:.2f}")
        print(f"Best Candidate:\n  {candidate_str}")

    def write_cache(self, filename=None):
        if filename is None:
            filename = 'driver_cache.pkl'

        cache = self.cache.copy()
        cache_info = self.cache_info.copy()

        with open(filename, 'wb') as f:
            pickle.dump((cache, cache_info), f)

    def read_cache(self, filename=None):
        if filename is None:
            filename = 'driver_cache.pkl'

        with open(filename, 'rb') as f:
            cache, cache_info = pickle.load(f)

        self.cache.update(cache)
        self.cache_info.update(cache_info)

    def wrapped_objective(self):
        """
        Update with new parallel structre TODO
        """
        @wraps(self.wrapped_objective)
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

                    if not isinstance(result, dict):
                        self.force_stop = True
                        self.check_interrupt()

                    result['caller'].append(name)
                    return result['objective']

                else:
                    # Result available in cache, no work needed
                    result['caller'].append(name)
                    return result['objective']

            except KeyError:
                # Candidate not in cache
                self.cache[candidate] = None  # indicates waiting in cache
                self.tasks.put((candidate, name))
                self.lock.release()
                self.cache_info['misses'] += 1

                # Poll cache for available result (should be threading.Condition)
                while (result := self.cache[candidate]) is None:
                    time.sleep(0.01)

                if not isinstance(result, dict):
                    self.force_stop = True
                    self.check_interrupt()

                if (self.best_obj is None) or (result['objective'] < self.best_obj):
                    self.best_obj = result['objective']
                    reason = 'new_best'

                else:
                    reason = ''

                info = dict(eval_time=result['eval_time'], reason=reason)
                self.eval_count += 1

                with self.lock:
                    self.print_log_line(info)

                self.cache_info['size'] += 1

                return result['objective']

        return wrapper


    def parallel_optimize(self, optimizers, opt_config, cache_file=None):
        # setup
        self.start_time = time.time()
        self.eval_count = 0
        self.force_stop = False

        # Establish communication queues and execution workers
        # Start workers
        n_opt = len(optimizers)
        num_workers = min(self.options['n_proc'], n_opt)

        self.init_parallel_workers(num_workers)

        if cache_file is not None:
            self.read_cache(cache_file)

        # Starting optimizer names
        self.opt_names = [opt.__name__ for opt in optimizers]
        obj = [partial(self.wrapped_objective(), name=name) for name in self.opt_names]
        opt = [partial(opt, **opt_config) for opt in optimizers]
        for i in range(n_opt):
            obj[i].__name__ = self.opt_names[i]

        self.print_log_header()

        with cf.ThreadPoolExecutor(max_workers=n_opt) as executor:
            try:
                threads = {executor.submit(opt[i], obj[i]):name for i,name in enumerate(self.opt_names)}

                for future in cf.as_completed(threads):
                    name = threads[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print(f"{name} generated an exception: {exc}")
                    else:
                        print(f"Optimizer {name} finished", data)

            except KeyboardInterrupt:
                pass

        # End worker processes
        self.cleanup_parallel()

        best_candidate, best_result = min(self.cache.items(), key=lambda item: item[1]['objective'])
        self.print_log_end(best_candidate, best_result['objective'])

        return best_candidate, best_result['objective']

    def parallel_execute(self, candidates, cache_file=None):
        # setup
        self.start_time = time.time()
        self.eval_count = 0
        self.force_stop = False

        # Establish communication queues and execution workers
        # Start workers
        num_workers = min(self.options['n_proc'], len(candidates))
        self.init_parallel_workers(num_workers)

        if cache_file is not None:
            self.read_cache(cache_file)

        self.opt_names = ['test']
        self.print_log_header()

        obj = [partial(self.wrapped_objective(), name=str(name)) for name in range(len(candidates))]
        for i in range(len(candidates)):
            obj[i].__name__ = str(i)

        with cf.ThreadPoolExecutor(max_workers=num_workers) as executor:
            try:
                threads = {executor.submit(obj[i], candidate): str(i) for i, candidate in enumerate(candidates)}

                for future in cf.as_completed(threads):
                    name = threads[future]
                    try:
                        data = future.result()

                    except Exception as exc:
                        print('%r generated an exception: %s' % (name, exc))

            except KeyboardInterrupt:
                pass

        # End worker processes
        self.cleanup_parallel()

        best_candidate, best_result = min(self.cache.items(), key=lambda item: item[1]['objective'])
        self.print_log_end(best_candidate, best_result['objective'])

        return best_candidate, best_result['objective']