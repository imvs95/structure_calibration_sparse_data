"""
Created on: 3-11-2021 12:35

@author: IvS
"""
import math
import copy

from platypus import Problem, Real, Integer

from utils.aggregate_statistics import aggregate_statistics

from pydsol.model.basic_logger import get_module_logger

logger = get_module_logger(__name__)


class GAProblem(Problem):
    """This class defines the problem for a Genetic Algorithm."""
    def __init__(
        self,
        decision_variables,
        ranges_variables,
        name_variables,
        obj_names,
        simulation_model,
        distance_metrics,
        df_ground_truth,
        multi_objective,
    ):
        super(GAProblem, self).__init__(
            nvars=1, nobjs=1, nconstrs=0
        )

        self.types[:] = [Integer(ranges_variables[0], ranges_variables[1])]
        self.directions[:] = Problem.MINIMIZE

        self.decision_variables = decision_variables
        self.decision_variables_names = name_variables
        self.obj_names = obj_names
        self.multi_objective = multi_objective

        self.simulation_model = simulation_model
        self.distance_metrics = distance_metrics
        self.df_ground_truth = df_ground_truth

    def evaluate(self, solution):
        """Evaluates the solution resulting from the GA. The parameters values are similar to the solution variables.
        The simulation model is runned for these parameters values, and a dataframe is returned.
        If there is one objective defined, only one objective is defined. If there are more objectives and
        multi-objective is True, then the objective values are calculated for each objectives seperatly.
        If there are more objectives and multi-objective is False, a sum of all objective values is taken as
        single objective value.

        Args:
            solution

        Returns:

        """
        par1 = solution.variables[0]

        graph_sol = self.decision_variables[par1]["graph"]
        results_sim_model, kpis_sim_model = self.simulation_model.run(parameters=[graph_sol])
        del graph_sol

        if self.nobjs == 1:
            # print("One objective")
            obj_dist = self.calculate_objectives(results_sim_model, par1)
            if isinstance(obj_dist, list):
                obj_dist = sum(obj_dist)

        elif (self.nobjs > 1) and (self.multi_objective is True):
            # print("Multiobjective")
            obj_dist = self.calculate_objectives(results_sim_model, par1)

        elif (self.nobjs > 1) and (self.multi_objective is False):
            # print("More objectives but mono-objective")
            all_obj_dist = self.calculate_objectives(results_sim_model, par1)
            obj_dist = sum(all_obj_dist)

        del results_sim_model, kpis_sim_model

        solution.objectives[:] = obj_dist

        solution.problem = copy.copy(solution.problem)
        solution.problem.decision_variables = {}
        solution.problem.simulation_model = {}
        solution.df_ground_truth = {}


    def calculate_objectives(self, result_sim_model, par1):
        """Calculates the distance per parameter between the df_in (ground truth) and the results of
        the simulation model, which is the objective. The distance is calculated by the given
        difference function. The distances are saved in a list.

        Args:
            result_sim_model (Dataframe): Results of the simulation model

        Returns:
            list: Distance per parameter, which is objective
        """
        obj_dist = []
        for obj in self.obj_names:
            try:
                dist = self.distance_metrics.calculate(
                    self.df_ground_truth[obj], result_sim_model[obj], debug=False
                )

                #normalize distance
                min_obj = self.df_ground_truth[obj]["p5"]
                max_obj = self.df_ground_truth[obj]["p95"]
                if min_obj == max_obj:
                    dist = min(dist, 1)
                else:
                    dist = (dist - min_obj)/(max_obj-min_obj)

                # print(
                #     "Solution: {0} {1:.2f} and {2} {3} with {4} gives {5}".format(
                #         self.decision_variables[0],
                #         par1,
                #         self.decision_variables[1],
                #         par2,
                #         obj,
                #         dist,
                #     )
                # )
                obj_dist.append(dist)
            except KeyError:
                continue
        return obj_dist
