from functools import wraps
from time import time
import operator

class OptimizationDriver():

    def __init__(self,
                 problem, #: OptimizationProblem,
                 optimizer, #: Callable[[np.array], float],
                 driver_kwargs=None,
                 optimizer_kwargs=None
                 ):

        self.problem = problem
        self.optimizer = optimizer
        self.driver_kwargs = driver_kwargs
        self.optimizer_kwargs = optimizer_kwargs
        self.objective = self.wrap_objective()
        self.cache = dict()
        self.start_time = None
        self.cache_info = {'hits': 0,
                           'misses': 0,
                           'size': 0,
                           'total_evals': 0}

        self.log_headers = ['Obj_Evals', 'Best_Objective', 'Eval_Time', 'Total_Time']
        self.log_widths = [len(header)+5 for header in self.log_headers]
        self.best_obj = None

    def wrap_objective(self):
        obj = self.problem.evaluate_objective

        if True: #scaling required?
            @wraps(obj)
            def wrapper(*args, **kwargs):
                candidate = self.problem.candidate_from_unit_array(*args)
                if self.start_time is None:
                    self.start_time = iter_start = time()
                    print("  ".join((val.rjust(width) for val, width in zip(self.log_headers, self.log_widths))))

                else:
                    iter_start = time()

                try:
                    self.cache_info['total_evals'] += 1
                    value = self.cache[candidate]
                    if value < self.best_obj:
                        self.best_obj = value
                    self.cache_info['hits'] += 1
                    log_values = [str(self.cache_info['total_evals']),
                                  f'{self.best_obj:8g}',
                                  f'*{time()-iter_start:.2f} sec',
                                  f'{time()-self.start_time:.2f} sec']
                    print("  ".join((val.rjust(width) for val, width in zip(log_values, self.log_widths))))
                    return value

                except:
                    self.cache_info['misses'] += 1

                elapsed = time() - self.start_time
                if elapsed > self.driver_kwargs['time_limit']:
                    print(f"Driver exiting, time limit: {self.driver_kwargs['time_limit']} secs exceeded")
                    raise Exception

                value = obj(candidate, **kwargs)
                if self.best_obj is None or value < self.best_obj:
                    self.best_obj = value
                self.cache[candidate] = value
                self.cache_info['size'] += 1
                log_values = [str(self.cache_info['total_evals']),
                              f'{self.best_obj:8g}',
                              f'{time() - iter_start:.2f} sec',
                              f'{time() - self.start_time:.2f} sec']
                print("  ".join((val.rjust(width) for val, width in zip(log_values, self.log_widths))))
                return value

            return wrapper

    def run(self):
        self.start_time = None
        try:
            u, v = self.optimizer(self.objective,  **self.optimizer_kwargs)

        except Exception:
            pass

        best_candidate, best_objective = min(self.cache.items(), key=operator.itemgetter(1))
        return best_candidate, best_objective