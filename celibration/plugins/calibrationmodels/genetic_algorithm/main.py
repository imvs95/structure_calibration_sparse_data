import pandas as pd
import numpy as np
import random
import json
import pickle
from celibration import (
    CalibrationModel,
    CalibrationModelFactory,
    CalibrationDifferenceFunction,
    CalibrationModelReport,
)
from typing import Any

from celibration.plugins.calibrationmodels.genetic_algorithm.problem import (
    GAProblem,
)
from celibration.plugins.calibrationmodels.genetic_algorithm.callback_convergence import (
    CallBack
)

from RunnerObject_generator import ComplexSupplyChainSimModel

from ema_workbench.em_framework.optimization import NSGAII, EpsNSGAII

from platypus import ProcessPoolEvaluator
from platypus import nondominated, CompoundOperator, SBX, HUX, PM, BitFlip


class PluginFactory(CalibrationModelFactory):
    def create_calibration_model(self, info=dict) -> CalibrationModel:
        return GeneticAlgorithmModel(info=info)


class GeneticAlgorithmModelReport(CalibrationModelReport):
    # self._model = model is set in the default constructor
    def get_string(self) -> str:
        missing_values = self._model.pluginscope["df_in"].isnull().sum().sum()
        result = (
            f"Report:\033[1m GeneticAlgorithmModelReport \033[0m:\n"
            # f"\tParameters:\n"
            # f"\t\t{self._kwargs}\n"
            f"\tModel:\n"
            f"\tDf_in missing: {missing_values} \n"
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
            "Results": self._model.pluginscope["solutions"],
            "Convergence": {"nfe": self._model.pluginscope["nfe_list"],
                            "eprogress": self._model.pluginscope["eprogress"]}
        }
        try:
            filename = filename.replace(".json", ".pkl")
            with open(filename, "wb") as f:
                pickle.dump(cal_model_json, f)

        except TypeError:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(json.dumps(cal_model_json))


class GeneticAlgorithmModel(CalibrationModel):
    def fit(
            self,
            df_in: pd.DataFrame,
            diff_func_object: CalibrationDifferenceFunction,
            diff_func_parameters: dict,
            debug: bool,
            **kwargs,
    ):
        """Performs the calibration via a Genetic Algorithm (GA).
        First, the problem is initialized. It can be multiobjective or not. Default is mono-objective.
        Next, the optimization is runned by an algorithm. Default algorithm used by GA is NSGAII.
        Other alternatives are epsNSGAII or BORG. It returns one optimal solution (in the case of mono-objective),
        or a Pareto front (in the case of multi-objective).

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

        if "multi_objective" in kwargs:
            multi_objective = kwargs["multi_objective"]
        else:
            multi_objective = False

        # these are required!
        try:
            self.pluginscope["decision_variables_names"] = kwargs["decision_variables_names"]
            # self.pluginscope["ground_truth_topology"] = kwargs["ground_truth_topology"]
        except KeyError:
            raise

        problem = self.initialize_problem(
            decision_variables=self._info["parameters"]["decision_variables"],
            ranges_variables=[list(self._info["parameters"]["decision_variables"].keys())[0],
                              list(self._info["parameters"]["decision_variables"].keys())[-1]],
            name_variables=kwargs["decision_variables_names"],
            obj_names=list(df_in.columns),
            simulation_model=ComplexSupplyChainSimModel,
            distance_metrics=diff_func_object,
            df_ground_truth=df_in,
            multi_objective=multi_objective,
        )

        variator = CompoundOperator(SBX(), HUX(), BitFlip())  # BitFlip() does not work without integer

        pool_processes = kwargs["num_pool"] if "num_pool" in kwargs else 1
        with ProcessPoolEvaluator(pool_processes) as evaluator:
            algorithm = NSGAII(
                problem,
                population_size=kwargs["population_size"],
                variator=variator,
                evaluator=evaluator
            )

            if "algorithm" in kwargs:
                if kwargs["algorithm"] == "NSGAII":
                    pass
                if kwargs["algorithm"].upper() == "EPSNSGAII":
                    algorithm = EpsNSGAII(
                        problem,
                        epsilons=kwargs["epsilons"],
                        population_size=kwargs["population_size"],
                        variator=variator,
                        evaluator=evaluator
                    )
            # nfe = population_size * generation
            # callback nfe only works when epsNSGAII
            callback = CallBack()
            algorithm.run(kwargs["nfe"], callback)

        solutions = self.get_solutions(algorithm)
        self.pluginscope["solutions"] = solutions
        # use pluginscope to use 'self' variables in other functions
        self.pluginscope["score"] = self.calculate_score_structural(
            solutions, self._info["parameters"]["ground_truth_topology"], self._info["parameters"]["decision_variables"],
            "betweenness")

        self.pluginscope["df_in"] = df_in

        self.pluginscope["nfe_list"] = callback.nfe
        self.pluginscope["eprogress"] = callback.eprogress

        return

    def initialize_problem(
            self,
            decision_variables: dict,
            ranges_variables: list,
            name_variables: list,
            obj_names: list,
            simulation_model: object,
            distance_metrics: CalibrationDifferenceFunction,
            df_ground_truth: pd.DataFrame,
            multi_objective: bool,
    ):
        """Initializes the problem.

        Args:
            decision_variables (list): list of decision variables
            ranges_variables (list): list of min and max value that a decision variable can take
            obj_names (list): list of the names of the objective columns
            simulation_model (object): Simulation Model Object (DSOL)
            distance_metrics (object): difference function
            df_ground_truth (Dataframe): df_in
            multi_objective (bool): True or False

        Returns:
            problem (object)
        """
        problem = GAProblem(
            decision_variables,
            ranges_variables,
            name_variables,
            obj_names,
            simulation_model,
            distance_metrics,
            df_ground_truth,
            multi_objective,
        )
        return problem

    def get_solutions(self, algorithm):
        """Transforms the solutions from the solution of GA to a readible values. It represents
        the decision variables of the solution and the objective of the nondominated solutions.

        Args:
            algorithm (object): algorithm of GA including results

        Returns:
            dict_all_solutions (dict)
        """
        dict_all_solutions = {}

        nondominated_solutions = nondominated(algorithm.result)

        for i in range(len(nondominated_solutions)):
            solution = nondominated_solutions[i]
            dict_all_solutions[i] = {}
            for var in range(algorithm.problem.nvars):
                decision_variable_name = algorithm.problem.decision_variables_names[var]
                var_value = algorithm.problem.types[var].decode(solution.variables[var])
                dict_all_solutions[i][decision_variable_name] = var_value

            if algorithm.problem.nobjs == 1:
                objective_name = "min_distance"
                obj_value = solution.objectives[0]
                dict_all_solutions[i][objective_name] = obj_value
            else:
                for obj in range(algorithm.problem.nobjs):
                    objective_name = algorithm.problem.obj_names[obj]
                    obj_value = solution.objectives[obj]
                    dict_all_solutions[i][objective_name] = obj_value

        return dict_all_solutions

    def normalize(self, x, min, max):
        return (x - min) / (max - min)

    def calculate_score_structural(self, solution, ground_truth_topology, decision_variables,
                                   topology):
        """Determine the score (quality of fit) based on the difference between graph (topology)
        and decision variable.

        Works only for single objective """
        sol_index = solution[0]["graph_structure"]
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
        df_solutions = pd.DataFrame.from_dict(
            self.pluginscope["solutions"], orient="index"
        ).drop_duplicates()
        dict_decision_variables = \
        df_solutions.iloc[:, :len(self.pluginscope["decision_variables_names"])].to_dict("records")[
            0]
        return dict_decision_variables, f"Summary CalibrationModel with solutions \n {df_solutions}" \
                                        f"\n and eprogress is {self.pluginscope['eprogress']}"

    def get_score(self):
        return self.pluginscope["score"]

    def get_report(self, **kwargs) -> CalibrationModelReport:
        report = GeneticAlgorithmModelReport(self, **kwargs)
        return report
