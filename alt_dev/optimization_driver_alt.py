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
    return reduce(lambda sub_dict, key: sub_dict.get(key, {}), keys, result)


class OptimizerInterrupt(Exception):
    """
    Stub exception used by the driver to interrupt optimizers (e.g., if time limit has been exceeded)
    """
    pass


class Worker(multiprocessing.Process):
    """
    Process-contained worker to execute objective calculations.
    """

    def __init__(self, task_queue, cache: dict, setup: Callable) -> None:
        """
        Process-contained worker, having an independent instance of the problem and simulation to evaluate the objective

        :param task_queue: multiprocessing.JoinableQueue()
        :param cache: multiprocessing.manager.dict()
        :param setup: function to create a new instance of the design problem
        """
        super().__init__()
        self.task_queue = task_queue
        self.cache = cache
        self.setup = setup

        logging.info("Worker process init")

    def run(self):
        """
        Poll the task queue until a task (candidate, caller_name) are available, (None, None) or a KeyboardInterrupt
            signals shutdown

        :return: None
        """
        logging.info("Worker process startup tasks")

        # Create a new problem for the worker
        problem = self.setup()
        # proc_name = self.name # not currently used
        candidate = None

        while True:
            try:
                # Get task from queue, this method blocks this process until a task is available
                candidate, caller_name = self.task_queue.get()
                logging.info(f"Worker process got {candidate} from queue")

                if candidate is None:
                    # Signal shutdown
                    logging.info(f"Worker process got None from queue, exiting")
                    self.task_queue.task_done()
                    break

                # Execute task, measure evaluation time
                start_time = time.time()
                candidate, result = problem.evaluate_objective(candidate)
                result['eval_time'] = time.time() - start_time
                result['caller'] = [caller_name]

            except KeyboardInterrupt:
                # Exit cleanly
                self.task_queue.task_done()

                # Signal any waiting optimizer threads to exit
                if candidate is not None:
                    self.cache[candidate] = OptimizerInterrupt

                logging.info(f"Worker process got KeyboardInterrupt, exiting")
                break

            # Objective returns normally, mark task as done and return result
            self.task_queue.task_done()
            self.cache[candidate] = result
            logging.info(f"Worker process calculated objective for {candidate}")


class OptimizationDriver():
    """
    Object to interface the HOPP optimization problem with humpday optimizers
    """
    DEFAULT_KWARGS = dict(time_limit=np.inf,
                          eval_limit=np.inf,
                          obj_limit=-np.inf,
                          n_proc=multiprocessing.cpu_count()-4,
                          log_freq=1,
                          log_file=None,
                          scaled=True)

    def __init__(self,
                 setup: Callable,
                 **kwargs) -> None:
        """
        Object to interface the HOPP optimization problem with humpday optimizers

        :param setup: Function which creates and returns a new instance of the optimization problem
        :param kwargs: Optional keyword arguments to change driver options (see DEFAULT_KWARGS)
        """
        logging.info("Driver init tasks")

        self.setup = setup
        self.problem = setup() # The driver needs an instance of the problem to access problem.candidate_from()
        self.parse_kwargs(kwargs)

        self.best_obj = None
        self.cache_info = dict(hits=0, misses=0, size=0, total_evals=0)
        self.get_candidate = self.problem.candidate_from_unit_array if self.options['scaled'] \
            else self.problem.candidate_from_array # Function to create formatted design candidates
        self.start_time = None
        self.force_stop = False
        self.eval_count = 0

    def parse_kwargs(self, kwargs: dict) -> None:
        """
        Helper function to set defaults and update options with user-provided input

        :param kwargs: Using **kwargs this is a dict of keyword arguments provided by the user
        :return: None
        """
        self.options = self.DEFAULT_KWARGS.copy()

        for key, value in kwargs.items():
            if key in self.options:
                self.options[key] = value
            else:
                print(f"Ignoring unknown driver option {key}={value}")

    def init_parallel_workers(self, num_workers: int) -> None:
        """
        Create the communication queue, cache dictionary, thread lock, and worker processes

        :param num_workers: Number of process-independent workers, which evaluate the objective.
        :return:
        """
        logging.info("Create parallel workers")

        self.tasks = multiprocessing.JoinableQueue()
        self.manager = multiprocessing.Manager()
        self.cache = self.manager.dict()
        self.lock = threading.Lock()

        print(f"Creating {num_workers} workers")
        self.workers = [Worker(self.tasks, self.cache, self.setup)
                           for _ in range(num_workers)]

        # Start the workers polling the task queue
        for w in self.workers:
            w.start()

        logging.info("Create parallel workers, done")

    def cleanup_parallel(self) -> None:
        """
        Cleanup all worker processes, signal them to exit cleanly, mark any pending tasks as complete

        :return: None
        """
        logging.info("Cleanup parallel workers")

        # If the driver receives a KeyboardInterrupt then the task queue needs to be emptied
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
            # Exit normally, None task signals each worker to exit
            for i in range(len(self.workers)):
                self.tasks.put((None, 'worker exit'))

        # Wait for all of the tasks to finish
        self.tasks.join()
        for w in self.workers:
            w.join()

        # Any tasks pending during a driver exception should be removed from the cache
        # Note that candidates producing simulation exceptions may still exist in the cache
        pop_list = []
        for key, value in self.cache.items():
            if not isinstance(value, dict):
                pop_list.append(key)

        _ = [self.cache.pop(key) for key in pop_list]

        logging.info("Cleanup tasks complete")

    def check_interrupt(self) -> None:
        """
        Check optional stopping criteria, these are specified by the user in the driver options

        :return: None
        """
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
        """
        Write driver cache out to pickle file

        :param filename: Optional path of file to write out the cache to
        :return:  None
        """
        if filename is None:
            filename = 'driver_cache.pkl'

        cache = self.cache.copy()
        cache_info = self.cache_info.copy()

        with open(filename, 'wb') as f:
            pickle.dump((cache, cache_info), f)

    def read_cache(self, filename=None):
        """
        Read the driver cache from file

        :param filename: Optional path of file to read the cache from
        :return:
        """
        if filename is None:
            filename = 'driver_cache.pkl'

        with open(filename, 'rb') as f:
            cache, cache_info = pickle.load(f)

        self.cache.update(cache)
        self.cache_info.update(cache_info)

    def wrapped_objective(self) -> None:
        """
        This method implements the logic to check if a candidate is in the cache, or is pending evaluation, or neither.
        Each optimizer thread needs its own copy of this method since they don't have access to the driver object, we
        can implement this by wrapping this method and returning the wrapped function. This allows the optimizer threads
        to share the driver object without explicitly passing it to them, and allows them to all use the shared task
        queue and driver cache.

        :return: None
        """
        eval_count = 0

        @wraps(self.wrapped_objective)
        def wrapper(*args, name=None, idx=None, objective_keys=None) -> float:
            """
            Objective function the optimizer threads call, assumes a parallel structure and avoids any re-calculations
                - Check if candidate is in cache, if so return objective stored in cache
                - If not, check if candidate is in queue (indicated by integer value in cache), wait for signal
                - If not, objective needs to be calculated, add candidate to task queue, poll cache for return,
                    and finally signal any threads waiting on the same candidate

            :param args: Follows the optimizer's convention of objective inputs (typically an array of floats)
            :param name: Caller name to insert into the result dictionary
            :param idx: Thread index, used for signal conditions
            :param objective_keys: Ordered list of keys to get the objective from the result dictionary
            :return: the numeric value being optimized
            """
            nonlocal eval_count
            eval_count += 1

            self.check_interrupt()
            candidate = self.get_candidate(*args)
            self.cache_info['total_evals'] += 1

            try:
                # Check if result in cache, throws KeyError if not
                self.lock.acquire()
                result = self.cache[candidate]
                self.lock.release()
                self.cache_info['hits'] += 1
                logging.info(f"Cache hit on candidate {candidate}")

                if isinstance(result, int):
                    # In cache but not complete, wait for complete signal
                    signal = self.conditions[result]
                    with signal:
                        signal.wait()

                    result = self.cache[candidate]

                    if not isinstance(result, dict):
                        self.force_stop = True
                        logging.info(f"Driver interrupt while waiting for objective evaluation")
                        self.check_interrupt()

                    # Append this caller name to the result dictionary
                    with self.lock:
                        result['caller'].append((name, eval_count))
                        self.cache[candidate] = result

                    logging.info(f"Cache wait returned on candidate {candidate}")
                    return recursive_get(result, objective_keys)

                else:
                    # Result available in cache, no work needed
                    # Append this caller name to the result dictionary
                    with self.lock:
                        result['caller'].append((name, eval_count))
                        self.cache[candidate] = result

                    logging.info(f"Cache hit returned on candidate {candidate}")
                    return recursive_get(result, objective_keys)

            except KeyError:
                # Candidate not in cache, nor waiting in queue
                self.cache[candidate] = idx  # indicates waiting condition for any other thread

                # Insert candidate and caller information into task queue
                self.tasks.put((candidate, (name, eval_count)))

                self.lock.release()
                self.cache_info['misses'] += 1
                logging.info(f"Cache miss on candidate {candidate}")

                # Poll cache for available result (unclear how this could be a threading.Condition signal)
                while isinstance(result := self.cache[candidate], int):
                    time.sleep(0.5)

                # Signal any other threads waiting on the same candidate
                signal = self.conditions[idx]
                with signal:
                    signal.notifyAll()

                # KeyboardInterrupt places a OptimizerInterrupt in the cache to signal a force_stop
                if not isinstance(result, dict):
                    self.force_stop = True
                    logging.info(f"Driver interrupt while waiting for objective evaluation")
                    self.check_interrupt()

                # Update best best objective if needed, and print a log line to console
                if (self.best_obj is None) or (result['net_present_values']['hybrid'] < self.best_obj):
                    self.best_obj = result['net_present_values']['hybrid']
                    reason = 'new_best'

                else:
                    reason = ''

                # additional information for log line
                info = dict(eval_time=result['eval_time'], reason=reason)

                with self.lock:
                    self.eval_count += 1
                    self.print_log_line(info)

                self.cache_info['size'] += 1
                logging.info(f"Cache new item returned on candidate {candidate}")
                return recursive_get(result, objective_keys)

        return wrapper


    def parallel_optimize(self, optimizers, opt_config, objective_keys, cache_file=None):
        """

        :param optimizers:
        :param opt_config:
        :param objective_keys:
        :param cache_file:
        :return:
        """
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
        """

        :param candidates:
        :param cache_file:
        :return:
        """
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

                    except OptimizerInterrupt:
                        pass

                    except Exception as exc:
                        print('%r generated an exception: %s' % (name, exc))

            except KeyboardInterrupt:
                pass

        # End worker processes
        self.cleanup_parallel()

        return len(candidates)