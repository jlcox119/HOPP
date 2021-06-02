import numpy as np
import traceback
from hybrid.hybrid_simulation import HybridSimulation


class HybridSizingProblem():  # OptimizationProblem (unwritten base)
    """
    Optimize the hybrid system sizing design variables
    """
    sep = '::'

    def __init__(self,
                 simulation: HybridSimulation,
                 design_variables: dict) -> None:
        """
        design_variables: dict of hybrid technologies each with a dict of design variable attributes
        """
        # super().__init__(simulation) missing base class
        self.simulation = simulation
        self.design_variables = design_variables
        self._check_design_variables()

    def _check_design_variables(self) -> None:
        """
        validate design_variables, bounds, prior
        """
        self.candidate_fields = [self.sep.join([tech, field])
                                 for tech,val in self.design_variables.items()
                                 for field,_ in val.items()]
        self.n_dim = len(self.candidate_fields)

        self._parse_bounds()
        self._validate_prior()

    def _parse_bounds(self) -> None:
        try:
            bounds = []

            for key, val in self.design_variables.items():
                for subkey, subval in val.items():
                    bounds.append(subval['bounds'])

                    assert len(bounds[-1]) == 2, \
                        f"{key}:{subkey} 'bounds' of length {len(bounds[-1])} not understood"

                    assert bounds[-1][0] <= bounds[-1][1], \
                        f"{key}:{subkey} invalid 'bounds': {bounds[-1][0]}(lower) > {bounds[-1][1]}(upper)"

            self.lower_bounds = np.array([bnd[0] for bnd in bounds])
            self.upper_bounds = np.array([bnd[1] for bnd in bounds])

        except KeyError as error:
            raise KeyError(f"{key}:{subkey} needs simple bounds defined as 'bounds':(lower,upper)") from error

    def _validate_prior(self) -> None:
        return
        priors = (0.5 * np.ones(self.candidate_idx[-1])) * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        i = 0

        for key, val in self.design_variables.items():
            for subkey, subval in val.items():
                try:
                    priors[i] = subval['prior']

                except KeyError:
                    # no prior given, assume midpoint
                    # print(f"{key}:{subkey} no 'prior' given")
                    pass

        candidate = self.candidate_from_array(priors)
        self._check_candidate(candidate)
        self._set_simulation_to_candidate(candidate)



    def _set_simulation_to_candidate(self,
                                     candidate: tuple) -> None:
        for field,value in candidate:
            tech_key, key = field.split(self.sep)
            tech_model = getattr(self.simulation, tech_key)

            if hasattr(tech_model, key):
                setattr(tech_model, key, value)
            else:
                tech_model.value(key, value)

    def _check_candidate(self,
                         candidate: tuple) -> None:
        """

        """
        return
        assert isinstance(candidate, self.DesignCandidate), \
            f"Design candidate must be a NamedTuple created with {self.__name__}.candidate...() methods"

        i = 0
        for field in candidate._fields:
            tech_vars = getattr(candidate, field)
            for subfield, value in zip(tech_vars._fields, tech_vars):
                assert (value >= self.lower_bounds[i]) and (value <= self.upper_bounds[i]), \
                    f"{field}:{subfield} invalid value ({value}), outside 'bounds':({self.lower_bounds[i]},{self.upper_bounds[i]})"
                i += 1

    def candidate_from_array(self, values: np.array) -> tuple:
        candidate = tuple([(field, val)
                           for field, val in zip(self.candidate_fields, values)])
        return candidate

    def candidate_from_unit_array(self, values: np.array) -> tuple:
        scaled_values = values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        candidate = tuple([(field, val)
                           for field,val in zip(self.candidate_fields, scaled_values)])
        return candidate

    def evaluate_objective(self, candidate: tuple) -> dict:
        result = dict()

        try:
            self._check_candidate(candidate)
            self._set_simulation_to_candidate(candidate)
            self.simulation.simulate(1)

            result['objective'] = self.simulation.net_present_values.hybrid

        except Exception:
            result['exception'] = traceback.format_exc()
            result['objective'] = np.nan

        return result