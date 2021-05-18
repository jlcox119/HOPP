from pathlib import Path

from collections import OrderedDict, namedtuple
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations
from hybrid.hybrid_simulation import HybridSimulation
from tools.optimization.optimization_problem_new import OptimizationProblem
import numpy as np


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
            self.ndim = len(bounds)

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
        assert isinstance(candidate, self.all_variables)

    def candidate_from_array(self, values: np.array):
        tech_candidates = [self.tech_variables[i](*values[idx[0]:idx[1]]) for i, idx in
                           enumerate(zip(self.candidate_idx[:-1], self.candidate_idx[1:]))]
        return self.all_variables(*tech_candidates)

    def evaluate_objective(self, candidate: namedtuple):
        self._check_candidate(candidate)

        self._set_simulation_to_candidate(candidate)
        self.simulation.simulate(1, is_test=True)
        evaluation = self.simulation.net_present_values.hybrid
        return evaluation


design_variables = OrderedDict(
    pv=      {'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
              'tilt':                {'bounds':(30,      60)}},
    battery= {'system_capacity_kwh': {'bounds':(150*1e3, 250*1e3)},
              'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
              'system_voltage_volts':{'bounds':(400,     600)}}
)

problem = HybridSizingProblem(hybrid_plant, design_variables)

# from humpday import OPTIMIZERS
#
# optimizer = cmaes(**config).run
#
#
# driver = OptimizationDriver(problem, optimizer, **kwargs)
#
# driver.run()

# pv = namedtuple('pv', ['system_capacity_kw', 'tilt'])
# pv_vars = pv(50*1e3, 45)
#
# battery = namedtuple('battery', ['system_capacity_kwh', 'system_capacity_kw', 'system_voltage_volts'])
# battery_vars = battery(200*1e3, 50*1e3, 500.0)
#
# Variables = namedtuple('Variables', ['pv', 'battery'])
#
# V = Variables(pv_vars, battery_vars)

# driver = OptimizationDriver(problem, optimizer, scaled=True)


"""
Occurs when creating the driver by passing in the problem
"""
import numpy as np
# move to problem init
tech_vars = [namedtuple(key, val.keys()) for key,val in design_variables.items()]
all_vars = namedtuple('Variables', design_variables.keys())


num_vars = [len(x._fields) for x in tech_vars]
num_vars_sum = np.concatenate(([0], np.cumsum(num_vars)))

nested_bounds = [[subval['bounds'] for subkey,subval in val.items()]
                     for key,val in design_variables.items()]

lower_bounds = np.array([item[0] for sublist in nested_bounds for item in sublist])
upper_bounds = np.array([item[1] for sublist in nested_bounds for item in sublist])


"""
Occurs when the optimizer needs to evaluate the objective
 driver wraps the problem objective function, as the optimizer provides vals, but the objective needs candidate
"""

vals = 0.25 * np.arange(5)
scaled_vals = vals * (upper_bounds - lower_bounds) + lower_bounds
candidate = problem.candidate_from_array(scaled_vals)
print(candidate)

problem.evaluate_objective(candidate)

