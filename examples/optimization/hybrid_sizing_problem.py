from pathlib import Path

from collections import OrderedDict, namedtuple
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations
from hybrid.hybrid_simulation import HybridSimulation
from tools.optimization.optimization_problem_new import OptimizationProblem

import numpy as np
import warnings
warnings.simplefilter("ignore")
import humpday
warnings.simplefilter("default")
from functools import wraps
from time import time
import operator

site = 'irregular'
location = locations[1]
site_data = None

if site == 'circular':
    site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
elif site == 'irregular':
    site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
else:
    raise Exception("Unknown site '" + site + "'")

g_file = Path(__file__).parent.parent.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"

site_info = SiteInfo(site_data, grid_resource_file=g_file)

# set up hybrid simulation with all the required parameters
solar_size_mw = 1
battery_capacity_mwh = 1
interconnection_size_mw = 150

technologies = technologies = {'pv': {
                    'system_capacity_kw': solar_size_mw * 1000,
                },
                # 'wind': {
                #     'num_turbines': 25,
                #     'turbine_rating_kw': 2000
                # },
                'battery': battery_capacity_mwh * 1000,
                'grid': interconnection_size_mw}

# Get resource

# Create model
dispatch_options = {'battery_dispatch': 'heuristic',
                    'n_look_ahead_periods': 24}
hybrid_plant = HybridSimulation(technologies,
                                site_info,
                                interconnect_kw=interconnection_size_mw * 1000,
                                dispatch_options=dispatch_options)

# Customize the hybrid plant assumptions here...
hybrid_plant.pv.value('inv_eff', 95.0)
hybrid_plant.pv.value('array_type', 0)

# Build a fixed dispatch array
#   length == n_look_ahead_periods
#   normalized (+) discharge (-) charge
fixed_dispatch = [0.0] * 6
fixed_dispatch.extend([-1.0] * 6)
fixed_dispatch.extend([1.0] * 6)
fixed_dispatch.extend([0.0] * 6)
# Set fixed dispatch
hybrid_plant.battery.dispatch.set_fixed_dispatch(fixed_dispatch)


class HybridSizingProblem(): #OptimizationProblem
    """
    Optimize the hybrid system sizing design variables
    """
    def __init__(self,
                 simulation: HybridSimulation,
                 design_variables: OrderedDict) -> None:
        """
        design_variables: dict of hybrid technologies each with a dict of design variable attributes
        """
        # super().__init__(simulation)
        self.simulation = simulation
        self._check_design_variables(design_variables)

    def _check_design_variables(self, design_variables: OrderedDict) -> None:
        """
        validate design_variables, bounds, prior
        """
        try:
            bounds = []

            for key, val in design_variables.items():
                for subkey, subval in val.items():
                    bounds.append(subval['bounds'])

                    assert len(bounds[-1]) == 2, \
                        f"{key}:{subkey} 'bounds' of length {len(bounds[-1])} not understood"

                    assert bounds[-1][0] <= bounds[-1][1], \
                        f"{key}:{subkey} invalid 'bounds': {bounds[-1][0]}(lower) > {bounds[-1][1]}(upper)"

            self.lower_bounds = np.array([bnd[0] for bnd in bounds])
            self.upper_bounds = np.array([bnd[1] for bnd in bounds])
            self.n_dim = len(bounds)

        except KeyError as error:
            raise KeyError(f"{key}:{subkey} needs simple bounds defined as 'bounds':(lower,upper)") from error

        # create candidate factory functions
        self.tech_variables = [namedtuple(key, val.keys()) for key,val in design_variables.items()]
        self.all_variables = namedtuple('DesignCandidate', design_variables.keys())

        num_vars = [len(x._fields) for x in self.tech_variables]
        self.candidate_idx = np.concatenate(([0], np.cumsum(num_vars)))

        priors = (0.5 * np.ones(self.candidate_idx[-1])) * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        i = 0

        for key, val in design_variables.items():
            for subkey, subval in val.items():
                try:
                    priors[i] = subval['prior']

                except KeyError:
                    # no prior given, assume midpoint
                    # print(f"{key}:{subkey} no 'prior' given")
                    pass

                assert (priors[i] >= self.lower_bounds[i]) and (priors[i] <= self.upper_bounds[i]), \
                    f"{key}:{subkey} invalid 'prior':{priors[i]}, outside 'bounds':({self.lower_bounds[i]},{self.upper_bounds[i]})"
                i += 1

        tech_candidates = [self.tech_variables[i](*priors[idx[0]:idx[1]]) for i, idx in
                            enumerate(zip(self.candidate_idx[:-1], self.candidate_idx[1:]))]
        candidate = self.all_variables(*tech_candidates)

        for tech_key in candidate._fields:
            tech_model = getattr(self.simulation, tech_key)
            tech_variables = getattr(candidate, tech_key)
            for key in tech_variables._fields:
                if hasattr(tech_model, key):
                    setattr(tech_model, key, getattr(tech_variables, key))
                else:
                    tech_model.value(key, getattr(tech_variables, key))

    def _set_simulation_to_candidate(self,
                                     candidate: namedtuple) -> None:
        for tech_key in candidate._fields:
            tech_model = getattr(self.simulation, tech_key)
            tech_variables = getattr(candidate, tech_key)
            for key in tech_variables._fields:
                if hasattr(tech_model, key):
                    setattr(tech_model, key, getattr(tech_variables, key))
                else:
                    tech_model.value(key, getattr(tech_variables, key))

    def _check_candidate(self, candidate: namedtuple):
        assert isinstance(candidate, self.all_variables), \
            f"Design candidate must be a NamedTuple created with {self.__name__}.candidate...() methods"

        i = 0
        for field in candidate._fields:
            tech_vars = getattr(candidate, field)
            for subfield, value in zip(tech_vars._fields, tech_vars):
                assert (value >= self.lower_bounds[i]) and (value <= self.upper_bounds[i]), \
                    f"{field}:{subfield} invalid value ({value}), outside 'bounds':({self.lower_bounds[i]},{self.upper_bounds[i]})"
                i += 1

    def candidate_from_array(self, values: np.array):
        tech_candidates = [self.tech_variables[i](*values[idx[0]:idx[1]]) for i, idx in
                           enumerate(zip(self.candidate_idx[:-1], self.candidate_idx[1:]))]
        return self.all_variables(*tech_candidates)

    def candidate_from_unit_array(self, values: np.array):
        scaled_values = values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        tech_candidates = [self.tech_variables[i](*scaled_values[idx[0]:idx[1]]) for i, idx in
                           enumerate(zip(self.candidate_idx[:-1], self.candidate_idx[1:]))]
        return self.all_variables(*tech_candidates)

    def evaluate_objective(self, candidate: namedtuple):
        self._check_candidate(candidate)

        self._set_simulation_to_candidate(candidate)
        self.simulation.simulate(1)
        evaluation = self.simulation.net_present_values.hybrid
        return evaluation

from collections.abc import Callable

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

design_variables = OrderedDict(
    pv=      {'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
              'tilt':                {'bounds':(30,      60)}},
    battery= {'system_capacity_kwh': {'bounds':(150*1e3, 250*1e3)},
              'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
              'system_voltage_volts':{'bounds':(400,     600)}}
)

problem = HybridSizingProblem(hybrid_plant, design_variables)

optimizer = humpday.OPTIMIZERS[1]

opt_config = dict(n_dim=problem.n_dim, n_trials=200, with_count=True)
driver_config = dict(time_limit=90)

driver = OptimizationDriver(problem, optimizer, optimizer_kwargs=opt_config, driver_kwargs=driver_config)
best_candidate, best_objective = driver.run()

print(best_objective)
print(best_candidate)

# from examples.optimization.hybrid_sizing_problem import *