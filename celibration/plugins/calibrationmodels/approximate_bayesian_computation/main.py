import pandas as pd
import numpy as np
import json
import random
import math
import pickle
from celibration import (
    CalibrationModel,
    CalibrationModelFactory,
    CalibrationDifferenceFunction,
    CalibrationModelReport,
)
from typing import Any

from celibration.plugins.calibrationmodels.approximate_bayesian_computation.mcmc_sampler import (
    adaptive_metropolis
)

from RunnerObject_generator import ComplexSupplyChainSimModel

from pydream.core import run_dream
from pydream.parameters import SampledParam
from pydream.convergence import Gelman_Rubin
from scipy.stats import uniform


# multiprocessing not working with python 3.10 -- check: https://youtrack.jetbrains.com/issue/PY-54447/BufferError-memoryview-has-1-exported-buffer

class PluginFactory(CalibrationModelFactory):
    def create_calibration_model(self, info=dict) -> CalibrationModel:
        return ABCModel(info=info)


class ABCReport(CalibrationModelReport):
    # self._model = model is set in the default constructor
    def get_string(self) -> str:
        result = (
            f"Report:\033[1m Approximate Bayesian Computation Model Report \033[0m:\n"
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

    def numpy_encoder(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError('Not serializable')

    def export_to_json(self, filename: str) -> Any:
        cal_model_json = {
            "Model_info": self._model.get_info(),
            "Score": self._model.pluginscope["score"],
            "Optimal_solution": self._model.pluginscope["min_solution"].to_dict(),
            "Acceptance_rate": self._model.pluginscope["acceptance_rate"],
            "Convergence": self._model.pluginscope["dict_convergence"],
            "Results": self._model.pluginscope["solutions"].to_dict(),
        }
        try:
            filename = filename.replace(".json", ".pkl")
            with open(filename, "wb") as f:
                pickle.dump(cal_model_json, f)

        except TypeError:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(cal_model_json))


class ABCModel(CalibrationModel):
    def fit(
            self,
            df_in: pd.DataFrame,
            diff_func_object: CalibrationDifferenceFunction,
            diff_func_parameters: dict,
            debug: bool,
            **kwargs,
    ):
        """Performs the calibration via the Approximate Bayesian Computation (ABC).
        First, it sets the default parameters, bounds, draws, and the likelihood function.
        Default algorithm used by ABC is Adapative Metropolis. It returns the results, scores and acceptance
        percentage. Hereafter, the solution with the smallest likelihood distance score is selected.

        Args:
            df_in (Dataframe): dataframe to fit on
            diff_func_object (object): difference function to use as likelihood
            kwargs: arguments in key-value format

        Returns:
            None
        """
        if "seed" in kwargs:
            random.seed(kwargs["seed"])
            np.random.seed(kwargs["seed"])
            # np.random.default_rng(kwargs["seed"])
            seed = kwargs["seed"]
        else:
            seed = None

        # self.pluginscope["decision_variables"] = kwargs["decision_variables"]
        self.pluginscope["decision_variables_names"] = kwargs["decision_variables_names"]
        self.pluginscope["df_ground_truth"] = df_in
        # self.pluginscope["ground_truth_topology"] = kwargs["ground_truth_topology"]

        self.pluginscope["objectives"] = kwargs["objectives"] if "objective" in kwargs else list(df_in.columns)
        bounds = np.asarray([[list(self._info["parameters"]["decision_variables"].keys())[0],
                              list(self._info["parameters"]["decision_variables"].keys())[-1]]])
        draws = kwargs["n_draws"]

        n_chains = kwargs["n_chains"] if "n_chains" in kwargs else 1

        convergence_progress = kwargs["convergence_progress"] if "convergence_progress" in kwargs else False

        if "algorithm" in kwargs:
            self.pluginscope["algorithm"] = kwargs["algorithm"]
            if kwargs["algorithm"].lower() == "pydream":
                # when multiple ranges -- multiple parameters with multiple SampledParam
                parameters_to_sample = SampledParam(
                    uniform, seed, loc=[bounds[0][0]], scale=[bounds[0][1] - 1])
                # for integer, no -1 is needed
                if convergence_progress:  # takes a long time
                    sampled_params, scores, acceptance_rate, conv_progress = self.run_dream_convergence_progress(
                        [parameters_to_sample], self.likelihood_distance, n_chains, niterations=draws, seed=seed)


                else:
                    sampled_params, scores, acceptance_rate = run_dream([parameters_to_sample],
                                                                        self.likelihood_distance,
                                                                        n_chains, niterations=draws, seed=seed)

                acceptance_rate = np.average(acceptance_rate)

                # Check convergence of multiple chains
                GR = Gelman_Rubin(sampled_params)
                if np.all(GR < 1.2):
                    converged = True
                else:
                    converged = False

                self.pluginscope["dict_convergence"] = {"total_iterations": draws,
                                                        "Gelman_Rubin": GR,
                                                        "converged": converged,
                                                        "progress": conv_progress if convergence_progress else None}

                print('Total iterations: ', draws, ' Gelman Rubin = ', GR, "converged = ", converged)
            else:
                pass
        else:
            self.pluginscope["algorithm"] = "adaptive_metropolis"
            # Default
            sampled_params, scores, acceptance_rate = adaptive_metropolis(
                self.likelihood_distance, bounds, draws
            )

        solutions, min_solution = self.get_solutions(sampled_params, scores, n_chains)
        self.pluginscope["solutions"] = solutions
        self.pluginscope["min_solution"] = min_solution

        self.pluginscope["acceptance_rate"] = acceptance_rate if "acceptance_rate" in vars() else 0

        self.pluginscope["score"] = self.calculate_score_structural(
            min_solution, self._info["parameters"]["ground_truth_topology"], self._info["parameters"]["decision_variables"],
            "betweenness")

        return

    def likelihood_distance(self, parameter_vector):
        """Calculates the distance for one or more objectives, based on the
        results of the simulation model and difference function.
        The results of the simulation model are calculated with the given parameters.
        If there is more than one objective, the distances between the df_in and
        the results of the simulation model are summed.


        Args:
            parameter_vector (Vector,array): Values of all parameters

        Returns:
            float: Returns the distance
        """
        # Run simulation
        try:
            # round because integer
            par1 = round(parameter_vector[0])
        except IndexError:
            par1 = round(parameter_vector[()])
        # par2 = round(parameter_vector[1])

        graph_sol = self._info["parameters"]["decision_variables"][par1]["graph"]
        results_sim_model, kpis_sim_model = ComplexSupplyChainSimModel.run(parameters=[graph_sol])
        del graph_sol

        # Calculate distance metrics
        if len(self.pluginscope["objectives"]) == 1:
            # print("One objective")
            dist = self.calculate_objectives(results_sim_model, par1)[0]

        else:
            # print("More objectives but mono-objective")
            all_obj_dist = self.calculate_objectives(results_sim_model, par1)
            dist = sum(all_obj_dist)

        if self.pluginscope["algorithm"].lower() == "pydream":
            dist = -dist

        del results_sim_model, kpis_sim_model

        return dist

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
                if min_obj == max_obj:
                    dist = min(dist, 1)
                else:
                    dist = (dist - min_obj) / (max_obj - min_obj)
                # print(
                #     "Solution: {0} {1:.2f} and {2} {3} with {4} gives {5}".format(
                #         self.pluginscope["decision_variables"][0],
                #         par1,
                #         self.pluginscope["decision_variables"][1],
                #         par2,
                #         obj,
                #         dist,
                #     )
                # )
                obj_dist.append(dist)
            except KeyError:
                continue
        return obj_dist

    def get_solutions(self, results, scores, n_chains):
        """Retrieves the solutions from the results and scores. It joins the results and scores
        dataframes resulting from the ABC. Hereafter, the values for the integer is changed
        (ABC uses it as a float). Next, it finds the solution with the minimum distance. This is
        the most optimal solution.

        Args:
            results (Dataframe): results from ABC
            scores (Dataframe): scores related to the results from ABC

        Returns:
            combined (Dataframe): all solutions resulting from ABC
            min_solution (Dataframe): solution with minimum score resulting from ABC
        """
        if n_chains > 1:
            df_results = pd.DataFrame()
            df_score = pd.DataFrame()
            for n in range(n_chains):
                df_chain_results = pd.DataFrame(results[n])
                df_results = pd.concat([df_results, df_chain_results])

                df_chain_score = abs(pd.DataFrame(scores[n]))
                df_score = pd.concat([df_score, df_chain_score])

            df_results.columns = self.pluginscope["decision_variables_names"]
            df_score.columns = ["Distance"]
            combined = df_results.reset_index(drop=True).join(df_score.reset_index(drop=True)). \
                drop_duplicates().reset_index(drop=True)


        else:
            df_results = pd.DataFrame(
                results, columns=self.pluginscope["decision_variables_names"]
            )
            df_score = pd.DataFrame(scores, columns=["Distance"])
            combined = df_results.join(df_score).drop_duplicates().reset_index(drop=True)

        # Find minimum value
        min_solution = combined[
            combined["Distance"] == min(combined["Distance"])
            ].reset_index(drop=True)

        return combined, min_solution

    def calculate_score(self, solution, ranges):
        """How close to the ground truth parameters"""
        min_par1 = ranges[0][0]
        max_par1 = ranges[0][1]
        # min_par2 = ranges[1][0]
        # max_par2 = ranges[1][1]

        gt_par1 = self.normalize(2.5, min_par1, max_par1)
        # gt_par2 = self.normalize(1, min_par2, max_par2)

        # Normalize the difference
        var_values = [
            solution[var][0] for var in self._info["parameters"]["decision_variables"]
        ]
        normalize_par1 = self.normalize(var_values[0], min_par1, max_par1)
        # normalize_par2 = self.normalize(var_values[1], min_par2, max_par2)

        diff_par1 = abs(gt_par1 - normalize_par1)
        # diff_par2 = abs(gt_par2 - normalize_par2)

        # Normalize together
        score = self.normalize(
            diff_par1, 0, len(self._info["parameters"]["decision_variables"])
        )  # (+diff_par_2)

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
        df_solutions = self.pluginscope["solutions"]
        min_solution = self.pluginscope["min_solution"]
        min_solution["round"] = round(min_solution[self.pluginscope["decision_variables_names"]])
        dict_decision_variables = \
        min_solution.iloc[:, :len(self.pluginscope["decision_variables_names"])].to_dict("records")[
            0]

        acceptance_rate = self.pluginscope["acceptance_rate"]
        return dict_decision_variables, (
            f"Summary CalibrationModel with solutions: \n {df_solutions} \n with the "
            f"most optimal solution: \n {min_solution} \n with an acceptance percentage "
            f"of {acceptance_rate * 100}%"
        )

    def get_score(self):
        # return self.pluginscope["min_solution"]["Score"][0]
        return self.pluginscope["score"]

    def get_report(self, **kwargs) -> CalibrationModelReport:
        report = ABCReport(self, **kwargs)
        return report

    def run_dream_convergence_progress(self, parameters, likelihood, nchains, niterations, **kwargs):
        """"This function runs DREAM and keeps track of the convergence via the Gelman-Rubin statistic
        at every 10% of the iterations."""

        n_iter_step = round(niterations * 0.1)  # 10%
        total_iterations = 0
        convergence = dict()

        # First iterations - no restart needed
        sampled_params, scores, acceptance_rate = run_dream(parameters, likelihood,
                                                            nchains, n_iter_step, seed=kwargs["seed"],
                                                            model_name="ivs")
        total_iterations += n_iter_step

        for chain in range(len(sampled_params)):
            np.save('ivs_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
                    sampled_params[chain])
            np.save('ivs_logps_chain_' + str(chain) + '_' + str(total_iterations),
                    scores[chain])

        GR = Gelman_Rubin(sampled_params)
        print('At iteration: ', total_iterations, ' GR = ', GR)
        np.savetxt('ivs_GelmanRubin_iteration_' + str(total_iterations) + '.txt', GR)

        if np.all(GR < 1.2):
            converged = True
        else:
            converged = False

        convergence[total_iterations] = {"GR": GR, "converged": converged}

        old_samples = sampled_params
        total_scores = scores
        total_acceptance_rate = np.array([[v] for v in acceptance_rate])

        starts = [sampled_params[chain][-1, :] for chain in range(nchains)]
        while total_iterations < niterations:
            sampled_params, scores, acceptance_rate = run_dream(parameters, likelihood,
                                                                nchains, n_iter_step, start=starts, restart=True,
                                                                seed=kwargs["seed"], model_name='ivs')
            total_iterations += n_iter_step

            for chain in range(len(sampled_params)):
                np.save('ivs_sampled_params_chain_' + str(chain) + '_' + str(total_iterations),
                        sampled_params[chain])
                np.save('ivs_logps_chain_' + str(chain) + '_' + str(total_iterations),
                        scores[chain])

            old_samples = [np.concatenate((old_samples[chain], sampled_params[chain])) for chain in range(nchains)]
            total_scores = [np.concatenate((total_scores[chain], scores[chain])) for chain in range(nchains)]
            total_acceptance_rate = [np.concatenate((total_acceptance_rate[chain],
                                                     np.array([[v] for v in acceptance_rate])[chain])) for chain in
                                     range(nchains)]

            # Check convergence of multiple chains
            GR = Gelman_Rubin(old_samples)
            if np.all(GR < 1.2):
                converged = True
            else:
                converged = False
            convergence[total_iterations] = {"GR": GR, "converged": converged}

            print('Iterations: ', total_iterations, ' Gelman Rubin = ', GR, "converged = ", converged)

            if (niterations - total_iterations) < n_iter_step:
                n_iter_step = (niterations - total_iterations)

        total_acceptance_rate = [np.average(ar) for ar in total_acceptance_rate]

        return old_samples, total_scores, total_acceptance_rate, convergence
