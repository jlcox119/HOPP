import numpy as np
from hybrid.hybrid_simulation import HybridSimulation
from collections import OrderedDict, namedtuple

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
        self.design_variables = design_variables
        self._check_design_variables()

    def _get_bounds(self) -> None:
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

    def _validate_prior(self):
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

    def _check_design_variables(self) -> None:
        """
        validate design_variables, bounds, prior
        """
        self.n_dim = len([key for sub in self.design_variables.values()
                              for key in sub.keys()])
        self._get_bounds()

        # create candidate factory functions
        self.tech_variables = [namedtuple(key, val.keys()) for key,val in self.design_variables.items()]
        self.all_variables = namedtuple('DesignCandidate', self.design_variables.keys())

        num_vars = [len(x._fields) for x in self.tech_variables]
        self.candidate_idx = np.concatenate(([0], np.cumsum(num_vars)))
        self._validate_prior()

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

        return -evaluation