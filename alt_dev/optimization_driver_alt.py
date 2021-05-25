from functools import wraps
from time import time
import operator

class OptimizerInterrupt(Exception):
    pass


class OptimizationDriver():

    def __init__(self,
                 problem, #: OptimizationProblem,
                 # optimizer, #: Callable[[np.array], float],
                 driver_kwargs=None,
                 optimizer_kwargs=None
                 ):

        self.problem = problem
        # self.optimizer = optimizer
        self.driver_kwargs = driver_kwargs
        self.optimizer_kwargs = optimizer_kwargs

        self.objective = self.wrap_objective()
        self.init_cache()
        self.get_candidate = self.problem.candidate_from_unit_array
        self.start_time = None

    def init_cache(self):
        self.best_obj = None
        self.cache = dict()
        self.cache_info = {'hits': 0,
                           'misses': 0,
                           'size': 0,
                           'total_evals': 0}

    def check_interrupt(self, reason):
        if reason == 'time_limit':
            self.iter_start = time()
            elapsed = self.iter_start- self.start_time
            if elapsed > self.driver_kwargs['time_limit']:
                print(f"Driver exiting, time limit: {self.driver_kwargs['time_limit']} secs")
                raise OptimizerInterrupt

        elif reason == 'eval_limit':
            if self.eval_count > self.driver_kwargs['eval_limit']:
                print(f"Driver exiting, eval limit: {self.driver_kwargs['eval_limit']}")
                raise OptimizerInterrupt

        elif reason == 'obj_limit':
            if (self.best_obj is not None) and (self.best_obj <= self.driver_kwargs['obj_limit']):
                print(f"Driver exiting, obj limit: {self.driver_kwargs['obj_limit']}")
                raise OptimizerInterrupt

    def print_log_header(self):
        self.log_headers = ['Obj_Evals', 'Best_Objective', 'Eval_Time', 'Total_Time']
        self.log_widths = [len(header) + 5 for header in self.log_headers]

        print()
        print('##### HOPP Optimization Driver #####'.center(sum(self.log_widths)))
        print('Driver Options:', self.driver_kwargs, sep='\n\t')
        print('Optimizer Options:', self.optimizer.__name__, self.optimizer_kwargs, sep='\n\t')
        print()
        print("".join((val.rjust(width) for val, width in zip(self.log_headers, self.log_widths))))

    def print_log_line(self, reason=None):
        if reason == 'cache_hit':
            prefix = 'c '

        elif reason == 'new_best':
            prefix = '* '

        else:
            prefix = ''

        curr_time = time()
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

    def get_from_cache(self, candidate):
        try:
            self.cache_info['total_evals'] += 1
            value = self.cache[candidate]
            self.cache_info['hits'] += 1
            self.print_log_line('cache_hit')
            return value

        except KeyError:
            self.eval_count += 1
            self.cache_info['misses'] += 1
            return None

    def add_to_cache(self, candidate, value):
        self.cache[candidate] = value
        self.cache_info['size'] += 1

    def wrap_objective(self):
        obj = self.problem.evaluate_objective

        if True: #scaling required?
            @wraps(obj)
            def wrapper(*args, **kwargs):
                self.check_interrupt('obj_limit')

                candidate = self.get_candidate(*args)

                if self.start_time is None:
                    self.start_time = time()
                    self.print_log_header()

                self.check_interrupt('time_limit')

                value = self.get_from_cache(candidate)
                if value is not None:
                    return value

                self.check_interrupt('eval_limit')

                value = obj(candidate, **kwargs)
                if self.best_obj is None or value < self.best_obj:
                    self.best_obj = value
                    self.print_log_line('new_best')

                else:
                    self.print_log_line()

                self.add_to_cache(candidate, value)

                return value

            return wrapper

    def run(self, optimizer):
        self.start_time = None
        self.eval_count = 0
        self.optimizer = optimizer

        try:
            optimizer(self.objective,  **self.optimizer_kwargs)

        except OptimizerInterrupt:
            pass

        best_candidate, best_objective = min(self.cache.items(), key=operator.itemgetter(1))
        self.print_log_end(best_candidate, best_objective)

        return best_candidate, best_objective