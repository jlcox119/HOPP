import time
import pickle
import queue
import traceback

import numpy as np
from functools import wraps, partial, reduce

import concurrent.futures as cf
import threading
import multiprocessing
import logging
from typing import Callable


def recursive_get(result: dict, keys: list) -> float:
    """
    Helper function for accessing a value in the nested result dictionary.
    Equivalent to result[keys[0]][keys[1]][keys[2]]...

    :param result: Simulation result nested dictionary
    :param keys: List of keys in order from highest to lowest level in the nested dictionary
    :return: Float value output from the simulation
    """
    print(type(result), keys)
    return reduce(lambda sub_dict, key: sub_dict.get(key, {}), keys, result)


class OptimizerInterrupt(Exception):
    """
    Stub exception used by the driver to interrupt optimizers (e.g., if time limit has been exceeded)
    """
    pass


class Worker(multiprocessing.Process):
    """
    Process worker to execute objective calculations.
    """

    def __init__(self, task_queue: object, cache: object, setup: Callable) -> None:

        super().__init__()
        self.task_queue = task_queue
        self.cache = cache
        self.setup = setup

        logging.info("Worker process init")

    def run(self):
        logging.info("Worker process startup tasks")

        # Create a new problem for the worker
        problem = self.setup()
        proc_name = self.name
        candidate = None

        while True:
            try:
                # Get task from queue
                candidate, name = self.task_queue.get()
                logging.info(f"Worker process got {candidate} from queue")

                if candidate is None:
                    # Signal shutdown
                    logging.info(f"Worker process got None from queue, exiting")
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

                logging.info(f"Worker process got KeyboardInterrupt, exiting")
                break

            # Mark task as done and return result
            self.task_queue.task_done()
            self.cache[candidate] = result
            logging.info(f"Worker process calculated objective for {candidate}")

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
        logging.info("Driver init tasks")

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
        logging.info("Create parallel workers")

        self.tasks = multiprocessing.JoinableQueue()
        self.manager = multiprocessing.Manager()
        self.cache = self.manager.dict()
        self.lock = threading.Lock()

        print(f"Creating {num_workers} workers")
        self.workers = [Worker(self.tasks, self.cache, self.setup)
                           for _ in range(num_workers)]

        for w in self.workers:
            w.start()

        logging.info("Create parallel workers, done")

    def cleanup_parallel(self):
        logging.info("Cleanup parallel workers")

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

        logging.info("Cleanup tasks complete")

    def check_interrupt(self):
        if self.force_stop:
            # print("Driver exiting, KeyBoardInterrupt")
            logging.info("Driver exiting, KeyBoardInterrupt")
            raise OptimizerInterrupt

        elapsed = time.time() - self.start_time
        if elapsed > self.options['time_limit']:
            # print(f"Driver exiting, time limit: {self.options['time_limit']} secs")
            logging.info(f"Driver exiting, time limit: {self.options['time_limit']} secs")
            raise OptimizerInterrupt

        if self.eval_count > self.options['eval_limit']:
            # print(f"Driver exiting, eval limit: {self.options['eval_limit']}")
            logging.info(f"Driver exiting, eval limit: {self.options['eval_limit']}")
            raise OptimizerInterrupt

        if (self.best_obj is not None) and (self.best_obj <= self.options['obj_limit']):
            # print(f"Driver exiting, obj limit: {self.options['obj_limit']}")
            logging.info(f"Driver exiting, obj limit: {self.options['obj_limit']}")
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
        Update with new parallel structure TODO
        Should probably explain why I'm doing this
        """
        eval_count = 0

        @wraps(self.wrapped_objective)
        def wrapper(*args, name=None, idx=None, objective_keys=None):
            nonlocal eval_count
            eval_count += 1

            self.check_interrupt()
            candidate = self.get_candidate(*args)
            self.cache_info['total_evals'] += 1

            try:
                # Check if result in cache
                self.lock.acquire()
                result = self.cache[candidate]
                self.lock.release()
                self.cache_info['hits'] += 1
                logging.info(f"Cache hit on candidate {candidate}")

                if isinstance(result, int):
                    # In cache but not complete, poll cache
                    # while (result := self.cache[candidate]) is None:
                    #     time.sleep(0.01)
                    signal = self.conditions[result]
                    with signal:
                        signal.wait()

                    result = self.cache[candidate]

                    if not isinstance(result, dict):
                        self.force_stop = True
                        logging.info(f"Driver interrupt while waiting for objective evaluation")
                        self.check_interrupt()

                    result['caller'].append((name, eval_count))
                    logging.info(f"Cache wait returned on candidate {candidate}")
                    return recursive_get(result, objective_keys)

                else:
                    # Result available in cache, no work needed
                    result['caller'].append((name, eval_count))
                    logging.info(f"Cache hit returned on candidate {candidate}")
                    return recursive_get(result, objective_keys)

            except KeyError:
                # Candidate not in cache
                self.cache[candidate] = idx  # indicates waiting condition

                self.tasks.put((candidate, (name, eval_count)))

                self.lock.release()
                self.cache_info['misses'] += 1
                logging.info(f"Cache miss on candidate {candidate}")

                # Poll cache for available result (should be threading.Condition)
                while isinstance(result := self.cache[candidate], int):
                    time.sleep(0.5)

                signal = self.conditions[idx]
                with signal:
                    signal.notifyAll()

                if not isinstance(result, dict):
                    self.force_stop = True
                    logging.info(f"Driver interrupt while waiting for objective evaluation")
                    self.check_interrupt()

                if (self.best_obj is None) or (result['net_present_values']['hybrid'] < self.best_obj):
                    self.best_obj = result['net_present_values']['hybrid']
                    reason = 'new_best'

                else:
                    reason = ''

                info = dict(eval_time=result['eval_time'], reason=reason)

                with self.lock:
                    self.eval_count += 1
                    self.print_log_line(info)

                self.cache_info['size'] += 1
                logging.info(f"Cache new item returned on candidate {candidate}")
                return recursive_get(result, objective_keys)

        return wrapper


    def parallel_optimize(self, optimizers, opt_config, objective_keys, cache_file=None):
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
        obj = [partial(self.wrapped_objective(), name=name, idx=i, objective_keys=objective_keys) for i,name in enumerate(self.opt_names)]
        opt = [partial(opt, **opt_config) for opt in optimizers]

        for i in range(n_opt):
            obj[i].__name__ = self.opt_names[i]

        self.conditions = [threading.Condition() for _ in range(n_opt)]

        self.print_log_header()

        with cf.ThreadPoolExecutor(max_workers=n_opt) as executor:
            try:
                threads = {executor.submit(opt[i], obj[i]):name for i,name in enumerate(self.opt_names)}

                for future in cf.as_completed(threads):
                    name = threads[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        err_str = traceback.format_exc()
                        print(f"{name} generated an exception: {err_str}")
                    else:
                        print(f"Optimizer {name} finished", data)

            except KeyboardInterrupt:
                pass

        # End worker processes
        self.cleanup_parallel()

        best_candidate, best_result = min(self.cache.items(), key=lambda item: recursive_get(item[1], objective_keys))
        self.print_log_end(best_candidate, recursive_get(best_result, objective_keys))

        return best_candidate, recursive_get(best_result, objective_keys)

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

        obj = [partial(self.wrapped_objective(), name=str(name), idx=name) for name in range(len(candidates))]
        for i in range(len(candidates)):
            obj[i].__name__ = str(i)

        self.conditions = [threading.Condition() for _ in range(len(candidates))]

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

        return len(candidates)