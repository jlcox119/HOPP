import time
import logging
import numpy as np
import traceback
from hybrid.hybrid_simulation import HybridSimulation
from pathlib import Path
from hybrid.sites import make_circular_site, make_irregular_site, SiteInfo, locations

SIMULATION_ATTRIBUTES = ['annual_energies', 'generation_profile', 'internal_rate_of_returns',
                         'lcoe_nom', 'lcoe_real', 'net_present_values', 'outputs_factory']

class HybridSizingProblem():  # OptimizationProblem (unwritten base)
    """
    Optimize the hybrid system sizing design variables
    """
    sep = '::'

    def __init__(self,
                 design_variables: dict) -> None:
        """
        design_variables: dict of hybrid technologies each with a dict of design variable attributes
        """
        # super().__init__(simulation) should not be done on init
        # self.simulation = simulation

        logging.info("Problem init")
        self.simulation = None
        self._parse_design_variables(design_variables)

    def _parse_design_variables(self,
                                design_variables: dict) -> None:
        """

        """
        self.design_variables = design_variables

        try:
            bounds = list()
            fields = list()
            field_set = set()
            precisions = list()

            for key, val in self.design_variables.items():
                for subkey, subval in val.items():

                    field_name = self.sep.join([key, subkey])
                    if field_name in field_set:
                        raise Exception(f"{field_name} repeated in design variables")

                    field_bounds = subval['bounds']
                    assert (num_bounds := len(field_bounds)) == 2, \
                        f"{key}:{subkey} 'bounds' of length {num_bounds} not understood"

                    assert field_bounds[0] <= field_bounds[1], \
                        f"{key}:{subkey} invalid 'bounds': {field_bounds[0]}(lower) > {field_bounds[1]}(upper)"

                    if 'precision' not in subval.keys():
                        precision = -6
                    else:
                        precision = subval['precision']

                    assert isinstance(precision, int), \
                        f"{key}:{subkey} invalid 'precision': {precision} must be an integer value"

                    precisions.append(precision)
                    field_set.add(field_name)
                    fields.append(field_name)
                    bounds.append(field_bounds)

            self.candidate_fields = fields
            self.n_dim = len(fields)
            self.lower_bounds = np.array([bnd[0] for bnd in bounds])
            self.upper_bounds = np.array([bnd[1] for bnd in bounds])
            self.precision = precisions

        except KeyError as error:
            raise KeyError(f"{key}:{subkey} needs simple bounds defined as 'bounds':(lower,upper)") from error

    def _check_candidate(self,
                         candidate: tuple) -> None:
        """
        ### Is this necessary?
        """
        assert (actual_length := len(candidate)) == self.n_dim, \
            f"Expected candidate with {self.n_dim} (field,value) pairs, got candidate of length {actual_length}"

        for i, (field, value) in enumerate(candidate):
            assert field == self.candidate_fields[i], \
                f"Expected field named {self.candidate_fields[i]} in position {i} of candidate, but found {field}"
            assert (value >= self.lower_bounds[i]) and (value <= self.upper_bounds[i]), \
                f"{field} invalid value ({value}), outside 'bounds':({self.lower_bounds[i]},{self.upper_bounds[i]})"

    def _set_simulation_to_candidate(self,
                                     candidate: tuple) -> None:
        for field,value in candidate:
            tech_key, key = field.split(self.sep)
            tech_model = getattr(self.simulation, tech_key)

            if hasattr(tech_model, key):
                setattr(tech_model, key, value)
            else:
                tech_model.value(key, value)

    def candidate_from_array(self, values: np.array) -> tuple:
        # assert that array is of the correct length?
        rounded_values = [np.round(x, decimals=-self.precision[i]) for i,x in enumerate(values)]
        candidate = tuple([(field, val)
                           for field, val in zip(self.candidate_fields, rounded_values)])
        return candidate

    def candidate_from_unit_array(self, values: np.array) -> tuple:
        # assert that array is of the correct length?
        scaled_values = values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        rounded_values = [np.round(x, decimals=-self.precision[i]) for i, x in enumerate(scaled_values)]
        candidate = tuple([(field, val)
                           for field,val in zip(self.candidate_fields, rounded_values)])
        return candidate

    def init_simulation(self):
        logging.info("Begin Simulation Init")

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

        technologies = {'pv': {'system_capacity_kw': solar_size_mw * 1000},
                        'battery': battery_capacity_mwh * 1000,
                        'grid': interconnection_size_mw}

        # Create model
        dispatch_options = {'battery_dispatch': 'heuristic', #simple
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
        logging.info("Simulation Init Complete")

        self.simulation = hybrid_plant

    def evaluate_objective(self, candidate: tuple) -> (tuple, dict):
        result = dict()
        if self.simulation is None:
            self.init_simulation()

        try:
            logging.info(f"Evaluating objective: {candidate}")

            self._check_candidate(candidate)
            self._set_simulation_to_candidate(candidate)
            self.simulation.simulate(1)

            tech_list = list(self.simulation.power_sources.keys()) + ['hybrid']
            for sim_output in SIMULATION_ATTRIBUTES:
                result[sim_output] = {key: value
                                      if not callable(value:=getattr(getattr(self.simulation, sim_output), key))
                                      else value()
                                      for key in tech_list}

            result['dispatch_factors'] = self.simulation.dispatch_factors

        except Exception:
            err_str = traceback.format_exc()
            logging.info(f"Error when evaluating objective: {err_str}")

            result['exception'] = err_str

        # raise KeyboardInterrupt

        logging.info(f"Objective evaluation complete: {candidate}")
        return candidate, result