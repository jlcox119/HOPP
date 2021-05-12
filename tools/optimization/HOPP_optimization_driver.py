from typing import (
    Optional,
    Type
)

from .data_logging.data_recorder import DataRecorder
from .optimization_problem_new import OptimizationProblem
from .driver.ask_tell_parallel_driver import AskTellParallelDriver
from .driver.ask_tell_serial_driver import AskTellSerialDriver
from .optimizer.CEM_optimizer import CEMOptimizer
from .optimizer.CMA_ES_optimizer import CMAESOptimizer
from .optimizer.GA_optimizer import GAOptimizer
from .optimizer.SPSA_optimizer import (
    SPSADimensionInfo,
    SPSAOptimizer,
)
from .optimizer.ask_tell_optimizer import AskTellOptimizer
from .optimizer.converting_optimization_driver_new import ConvertingOptimizationDriver
from .optimizer.dimension.gaussian_dimension import Gaussian
from .optimizer.stationary_optimizer import StationaryOptimizer


class HOPPOptimizationDriver(ConvertingOptimizationDriver):
    """
    Creates a ConvertingOptimizationDriver with the given optimizer method and initial conditions from the
    problem's prior
    """

    def __init__(self,
                 problem: OptimizationProblem,
                 method: str,
                 recorder: DataRecorder,
                 nprocs: Optional[int] = None,
                 **kwargs
                 ) -> None:
        self.problem: OptimizationProblem = problem

        optimizer: AskTellOptimizer
        prior: object
        if method == 'GA':
            optimizer, prior = self.genetic_algorithm(**kwargs)
        elif method == 'CEM':
            optimizer, prior = self.cross_entropy(**kwargs)
        elif method == 'CMA-ES':
            optimizer, prior = self.CMA_ES(**kwargs)
        elif method == 'SPSA':
            optimizer, prior = self.genetic_algorithm(**kwargs)
        elif method == 'Stationary':
            optimizer, prior = self.stationary_optimizer(**kwargs)
        else:
            raise ValueError('Unknown optimizer: "' + method + '"')

        driver = AskTellSerialDriver() if nprocs == 1 else AskTellParallelDriver(nprocs)
        super().__init__(
            driver,
            optimizer,
            # ObjectConverter(),
            prior,
            conformer=self.problem.conform_candidate_and_get_penalty,
            objective=self.problem.objective,
            recorder=recorder,
        )

    @staticmethod
    def check_kwargs(inputs: tuple,
                     **kwargs
                     ) -> None:
        """
        Checks that the inputs are in **kwargs
        :param inputs: tuple of strings
        :param kwargs: keyword arguments
        """
        for i in inputs:
            if i not in kwargs:
                raise ValueError(i + " argument required")

    def create_prior(self,
                     dimension_type: type,
                     conf_prior_params: dict = None
                     ):
        """
        Create a prior candidate with information about each dimension's distribution
        :param dimension_type: the distribution type of each dimension of the prior
        :param conf_prior_params: a nested dictionary containing key: value pairs where the key is the dimension name
                    and the value is a dictionary of the distributions' parameters that should replace the default ones
                    that are stored in the problem's get_prior_params function. Parameters that are not attributes of
                    dimension_type are not used

        Example:
            config_prior_params = { "border_spacing": {"mu": 3, "beta": 4}}
                This will replace the prior's border_spacing distribution's mu parameter to be 3, but beta is ignored
        :return:
        """
        prior_params = self.problem.get_prior_params(dimension_type)

        if conf_prior_params:
            for conf_dimension in conf_prior_params.keys():
                if conf_dimension in prior_params.keys():
                    prior_params[conf_dimension].update({k: v for k, v in conf_prior_params[conf_dimension].items()
                                                         if k in prior_params[conf_dimension].keys()})

        # for conf_dimension, v in prior_params.items():
        #     prior_params[conf_dimension] = dimension_type(**v)

        # self.problem.convert_to_candidate(prior_params)
        return list(dimension_type(**v) for _, v in prior_params.items())

    def cross_entropy(self,
                      **kwargs
                      ) -> (AskTellOptimizer, object):
        """
        Create a cross entropy optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = CEMOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior

    def CMA_ES(self,
               **kwargs
               ) -> (AskTellOptimizer, object):
        """
        Create a cross entropy optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = CMAESOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior

    def genetic_algorithm(self,
                          **kwargs
                          ) -> (AskTellOptimizer, object):
        """
        Create a genetic algorithm optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = GAOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior

    def simultaneous_perturbation_stochastic_approximation(self,
                                                           **kwargs
                                                           ) -> (AskTellOptimizer, object):
        """
        Create a SPSA optimizer using a SPSA sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(SPSADimensionInfo, kwargs.get("prior_params"))
        optimizer = SPSAOptimizer(.2, num_estimates=kwargs.get("generation_size"))
        return optimizer, prior

    def stationary_optimizer(self,
                             **kwargs
                             ) -> (AskTellOptimizer, object):
        """
        Create a stationary optimizer using a Gaussian sampling distribution with required keyword arguments:
            prior_scale: float = scaling factor
            generation_size: int = number of candidates to generate
            selection_proportion: float = proportion of candidates
        :param kwargs: keyword arguments
        :return: optimizer, prior
        """
        args = ("prior_scale", "generation_size", "selection_proportion")
        self.check_kwargs(args, **kwargs)

        prior = self.create_prior(Gaussian, kwargs.get("prior_params"))
        optimizer = StationaryOptimizer(kwargs.get("generation_size"), kwargs.get("selection_proportion"))
        return optimizer, prior