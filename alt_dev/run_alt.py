
from alt_dev.optimization_problem_alt import HybridSizingProblem
from alt_dev.optimization_driver_alt import OptimizationDriver

import warnings
warnings.simplefilter("ignore")
import humpday
warnings.simplefilter("default")

import pyDOE


def problem_setup():
    # Define Design Optimization Variables
    design_variables = dict(
        pv=      {'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
                  'tilt':                {'bounds':(30,      60)}},
        battery= {'system_capacity_kwh': {'bounds':(150*1e3, 250*1e3)},
                  'system_capacity_kw':  {'bounds':(25*1e3,  75*1e3)},
                  'system_voltage_volts':{'bounds':(400,     600)}}
    )

    # Problem definition
    problem = HybridSizingProblem(design_variables)

    return problem


if __name__ == '__main__':

    # Driver config
    driver_config = dict(eval_limit=200, obj_limit=-3e8, n_proc=6) #time_limit=40,

    # Driver init
    driver = OptimizationDriver(problem_setup, **driver_config)



    # # Optimizer callable init
    # optimizers = humpday.OPTIMIZERS[:5]
    # opt_config = dict(n_dim=5, n_trials=50, with_count=True)
    #
    # # Call all optimizers in parallel
    # best_candidate, best_objective = driver.parallel_optimize(optimizers, opt_config)#, cache_file='driver_cache.pkl')


    # Get experiment candidates, and evaluate objective in parallel
    candidates = pyDOE.lhs(5, criterion='center', samples=24)
    best_candidate, best_objective = driver.parallel_execute(candidates)

    # Check on the driver cache
    print(driver.cache_info)