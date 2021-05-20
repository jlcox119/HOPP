from pathlib import Path
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations

from hybrid.hybrid_simulation import HybridSimulation
from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver

import warnings
warnings.simplefilter("ignore")
import humpday
warnings.simplefilter("default")

from collections import OrderedDict



site = 'irregular'
location = locations[1]
site_data = None

if site == 'circular':
    site_data = make_circular_site(lat=location[0], lon=location[1], elev=location[2])
elif site == 'irregular':
    site_data = make_irregular_site(lat=location[0], lon=location[1], elev=location[2])
else:
    raise Exception("Unknown site '" + site + "'")

g_file = Path(__file__).parent.parent / "resource_files" / "grid" / "pricing-data-2015-IronMtn-002_factors.csv"

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

# Define Design Optimization Variables
design_variables = OrderedDict(
    pv=      {'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
              'tilt':                {'bounds':(30,      60)}},
    battery= {'system_capacity_kwh': {'bounds':(150*1e3, 250*1e3)},
              'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
              'system_voltage_volts':{'bounds':(400,     600)}}
)

# Problem definition
problem = HybridSizingProblem(hybrid_plant, design_variables)

# Optimizer callable init
optimizer = humpday.OPTIMIZERS[1]

# Optimizer and driver config
opt_config = dict(n_dim=problem.n_dim, n_trials=200, with_count=True)
driver_config = dict(time_limit=20)

# Driver init
driver = OptimizationDriver(problem,
                            optimizer,
                            optimizer_kwargs=opt_config,
                            driver_kwargs=driver_config)

best_candidate, best_objective = driver.run()


# from examples.optimization.hybrid_sizing_problem import *