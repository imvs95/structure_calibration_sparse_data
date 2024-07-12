import pandas as pd
import numpy as np
import random
import json
import pickle
import multiprocessing as mp

from bayes_opt import BayesianOptimization

from celibration import (
    CalibrationModel,
    CalibrationModelFactory,
    CalibrationDifferenceFunction,
    CalibrationModelReport,
)
from typing import Any

from RunnerObject_generator_mp import ComplexSupplyChainSimModel


class PluginFactory(CalibrationModelFactory):
    def create_calibration_model(self, info=dict) -> CalibrationModel:
        return BayesianOptimizationModel(info=info)


class BayesianOptimizationReport(CalibrationModelReport):
    # self._model = model is set in the default constructor
    def get_string(self) -> str:
        result = (
            f"Report:\033[1m BayesianOptimizationReport \033[0m:\n"
            # f"\tParameters:\n"
            # f"\t\t{self._kwargs}\n"
            f"\tModel:\n"
            f"\t\tResults: {self._model.get_summary()[1]}\n"
        ).expandtabs(2)
        return result

    def render(self, type: str) -> Any:
        return NotImplementedError

    def export_to_file(self, type: str, filename: str) -> Any:
        if type == "pdf":
            self.export_to_pdf(filename=filename)
        if type == "json":
            self.export_to_json(filename=filename)
        return NotImplementedError

    def export_to_json(self, filename: str) -> Any:
        cal_model_json = {
            "Model_info": self._model.get_info(),
            "Score": self._model.pluginscope["score"],
            "Results": self._model.pluginscope["solutions"].to_dict(),
            "Convergence": {"n_iterations": self._model.pluginscope["n_iterations"],
                            "improvements": self._model.pluginscope["n_improvements"],
                            "sum_obj": self._model.pluginscope["conv_sum_obj"],
                            "avg_sum_obj": [a / b for a, b in zip(self._model.pluginscope["conv_sum_obj"][1:],
                                                  self._model.pluginscope["n_iterations"][1:])]
                            }
        }

        try:
            filename = filename.replace(".json", ".pkl")
            with open(filename, "wb") as f:
                pickle.dump(cal_model_json, f)

        except TypeError:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(cal_model_json))


class BayesianOptimizationModel(CalibrationModel):
    def fit(
            self,
            df_in: pd.DataFrame,
            diff_func_object: CalibrationDifferenceFunction,
            diff_func_parameters: dict,
            debug: bool,
            **kwargs,
    ):
        """Performs the calibration via the Bayesian Optimization. This will be 
        further elaborated in the documentation per functions. 

        Args:
            df_in (Dataframe): dataframe to fit on
            diff_func_object (object): difference function to use as likelihood
            kwargs: arguments in key-value format
        """
        if "seed" in kwargs:
            random.seed(kwargs["seed"])
            np.random.seed(kwargs["seed"])
            self.pluginscope["seed"] = kwargs["seed"]

        self.pluginscope["df_ground_truth"] = df_in
        # self.pluginscope["ground_truth_topology"] = kwargs["ground_truth_topology"]

        # self.pluginscope["decision_variables"] = kwargs["decision_variables"]
        self.pluginscope["decision_variables_names"] = kwargs["decision_variables_names"]
        self.pluginscope["objectives"] = kwargs["objectives"] if "objective" in kwargs else list(df_in.columns)
        self.pluginscope["ranges_variables"] = [[list(self._info["parameters"]["decision_variables"].keys())[0],
                                                 list(self._info["parameters"]["decision_variables"].keys())[-1]]]

        self.pluginscope["initial_points"] = kwargs["initial_points"]
        self.pluginscope["n_iterations"] = kwargs["n_iterations"]
        if "exploration_strategy" in kwargs:
            self.pluginscope["exploration_strategy"] = kwargs["exploration_strategy"]
        else:
            self.pluginscope["exploration_strategy"] = "ucb"
            # Default is UBC strategy

        # self.pluginscope["multi_objective"] = kwargs["multi_objective"]

        pool_processes = kwargs["num_pool"] if "num_pool" in kwargs else 1

        with mp.Pool(processes=pool_processes) as pool:
            self.pluginscope["pool"] = pool
            results = self.bayesian_optimization()

        df_solutions, objective = self.get_solutions(results)

        del results
        del self.pluginscope["pool"]

        self.pluginscope["score"] = self.calculate_score_structural(df_solutions,
                                                                    self._info["parameters"]["ground_truth_topology"],
                                                                    self._info["parameters"]["decision_variables"],
                                                                    "betweenness")  # score
        self.pluginscope["solutions"] = df_solutions

        return

    def distance_function(self, par1):
        """Calculates the distance for one or more objectives, based on the 
        results of the simulation model and difference function. The results of
        the simulation model are calculated with the given parameters.
        If there is more than one objective, the distances between the df_in and
        the results of the simulation model are summed.

        Args:
            par1 (float): Value of parameter 1

        Returns:
            float: Returns the negative value of the distance, which is the objective.
            The negative objective is needed for bayesian optimisation.
        """
        par1 = round(par1)
        # par2 = round(parameter_vector[1])

        graph_sol = self._info["parameters"]["decision_variables"][par1]["graph"]
        results_sim_model, kpis_sim_model = ComplexSupplyChainSimModel.run(parameters=[graph_sol],
                                                                           pool_obj=self.pluginscope["pool"])
        del graph_sol

        # Calculate distance metrics
        if len(self.pluginscope["objectives"]) == 1:
            # print("One objective")
            dist = self.calculate_objectives(results_sim_model)[0]

        elif len(self.pluginscope["objectives"]) > 1:
            # print("More objectives but mono-objective")
            all_obj_dist = self.calculate_objectives(results_sim_model)
            dist = sum(all_obj_dist)

        del results_sim_model, kpis_sim_model
        # When optimizing, it is only possible to take a maximum, hence -1*dist
        return -dist

    def calculate_objectives(self, result_sim_model):
        """Calculates the distance per parameter between the ground truth and the results of 
        the simulation model, which is the objective. The distance is calculated by the given 
        difference function(s).

        Args:
            result_sim_model (Dataframe): Results of the simulation model

        Returns:
            list: Distance per parameter, which is objective
        """
        obj_dist = []
        for obj in self.pluginscope["objectives"]:
            try:
                dist = self.diff_func_object.calculate(
                    self.pluginscope["df_ground_truth"][obj],
                    result_sim_model[obj],
                    debug=False,
                )

                # normalize distance
                min_obj = self.pluginscope["df_ground_truth"][obj]["p5"]
                max_obj = self.pluginscope["df_ground_truth"][obj]["p95"]
                dist = (dist - min_obj) / (max_obj - min_obj)
                # print(
                #     "Solution: {0} {1:.2f} and {2} {3} with {4} gives {5}".format(
                #         self.pluginscope["decision_variables"][0],
                #         par1,
                #         self.pluginscope["decision_variables"][1],
                #         par2,
                #         obj,
                #         -dist,
                #     )
                # )
                obj_dist.append(dist)
            except KeyError:
                continue
        return obj_dist

    def bayesian_optimization(self):
        """The bayesian optimisation calculates the maximum of the distance function, 
        the parameter bounds are the given input. Maximising the distance function 
        gives the minimum distance between the simulation model and the ground true. 
        As a result, it is known at which parameters the simulation model performs best 
        and the minimum distance, which indicates how well the optimisation model performs. 
        
        Returns:
            dict: All results of the bayesian optimisation
        """
        variable1 = tuple(self.pluginscope["ranges_variables"][0])
        # variable2 = self.pluginscope["ranges_variables"][1]
        optimization_bounds = {
            "par1": variable1,
        }

        optimizer = BayesianOptimization(
            f=self.distance_function,
            pbounds=optimization_bounds,
            random_state=self.pluginscope["seed"],
        )

        optimizer.maximize(
            init_points=self.pluginscope[
                "initial_points"
            ],  # steps of random exploration, helps by diversifying the exploration space
            n_iter=self.pluginscope[
                "n_iterations"
            ],  # steps of bayesian optimization, more steps more likely to find a good maximum
            acq=self.pluginscope["exploration_strategy"],
            # exploration strategy: ubc for Upper Confidence Bound, or ei for Expected Improvement, or
            # poi for Probability of Improvement
        )

        results = optimizer

        self.get_improvement_convergence(results)

        return results

    def get_improvement_convergence(self, results):
        all_results = results.res
        num_iterations = [0, 1]
        num_improvement = [0, 1]
        first_res = all_results[0]["target"]
        sum_obj_values = [0, first_res]

        obj_value = first_res
        remaining_res = all_results[1:]
        for idx, res in zip(range(len(remaining_res)), remaining_res):
            num_iterations.append(idx + 2)
            new_res = res["target"]
            obj_value += new_res
            sum_obj_values.append(obj_value)

            if first_res < new_res:
                num_improvement.append(1)
            else:
                num_improvement.append(0)
            first_res = new_res

        self.pluginscope["n_iterations"] = num_iterations
        self.pluginscope["n_improvements"] = num_improvement
        self.pluginscope["conv_sum_obj"] = sum_obj_values

        del num_iterations, num_improvement, sum_obj_values

    def get_solutions(self, results):
        """The results are received by the bayesian optimisation.The score is the maximum result
        of the BO model, where the negative is taken because the BO can only maximise. The values 
        of the parameters corresponding to the maximum results are converted into the solution 
        dataframe. Parameter 2 is converted to an integer. The bayesian optimisation gives the 
        solution that will approach the global maximum.

        Args:
            results (dict): Results of bayesian optimisation

        Returns:
            dataframe: Associated parameters with which the bayesian optimisation is optimised.
            float: The minimum distance, which indicates how well the optimisation model performs.

        """
        # df_all_results = pd.DataFrame.from_dict(results.res)
        results_parameters = []

        for key, value in results.max["params"].items():
            if key == "par1":
                results_parameters.append(round(value))
            else:
                results_parameters.append(value)

        # Reverse -1*dist
        score = -results.max["target"]

        df_results = pd.DataFrame(
            [results_parameters], columns=self.pluginscope["decision_variables_names"]
        )
        df_score = pd.DataFrame([score], columns=["min_distance"])
        df_solutions = (
            df_results.join(df_score).drop_duplicates().reset_index(drop=True)
        )

        return df_solutions, score

    def calculate_score(self, solution, ranges):
        """How close to the ground truth parameters"""
        min_par1 = ranges[0][0]
        max_par1 = ranges[0][1]
        # min_par2 = ranges[1][0]
        # max_par2 = ranges[1][1]

        gt_par1 = self.normalize(2.5, min_par1, max_par1)
        # gt_par2 = self.normalize(1, min_par2, max_par2)

        # Normalize the difference
        var_values = [solution[var][0] for var in self._info["parameters"]["decision_variables"]]
        normalize_par1 = self.normalize(var_values[0], min_par1, max_par1)
        # normalize_par2 = self.normalize(var_values[1], min_par2, max_par2)

        diff_par1 = abs(gt_par1 - normalize_par1)
        # diff_par2 = abs(gt_par2 - normalize_par2)

        # Normalize together
        score = self.normalize(diff_par1, 0, len(self._info["parameters"]["decision_variables"]))
        # (+diff_par_2)

        # The higher the score, the closer it is to the real parameter
        return 1 - score

    def normalize(self, x, min, max):
        return (x - min) / (max - min)

    def calculate_score_structural(self, solution, ground_truth_topology, decision_variables,
                                   topology):
        """Determine the score (quality of fit) based on the difference between graph (topology)
        and decision variable.

        Works only for single objective """
        if len(solution) > 1:
            sol_round = [round(r[1]["graph_structure"]) for r in solution.iterrows()]
            unique_round = np.unique(sol_round)
            try:
                sol_index = unique_round[0]
                assert isinstance(sol_index, (int, np.uint, np.integer))
            except AssertionError:
                sol_index = unique_round[0]
                print("No unique minimum solution is found, we have {0} and we choose {1}".format(unique_round,
                                                                                                  sol_index))
        else:
            sol_index = round([
                                  solution[var][0] for var in self.pluginscope["decision_variables_names"]
                              ][0])

        info_solution_graph = decision_variables[sol_index]

        ground_truth_graph = ground_truth_topology["graph"][0]
        solution_graph = info_solution_graph["graph"]

        # score on topology
        gt_value = ground_truth_topology[topology][0]
        sol_value = info_solution_graph[topology]
        diff_topology = abs(gt_value - sol_value)

        # The higher the score, the closer it is to the real parameter
        return 1 - diff_topology

    def get_summary(self):
        dict_decision_variables = \
            self.pluginscope["solutions"].iloc[:, :len(self.pluginscope["decision_variables_names"])].to_dict(
                "records")[0]
        df_results = self.pluginscope["solutions"]
        df_results["round"] = round(df_results[self.pluginscope["decision_variables_names"]])
        return dict_decision_variables, f"Summary CalibrationModel with most optimal solution:  \n {df_results}"

    def get_score(self):
        return self.pluginscope["score"]

    def get_report(self, **kwargs) -> CalibrationModelReport:
        report = BayesianOptimizationReport(self, **kwargs)
        return report
