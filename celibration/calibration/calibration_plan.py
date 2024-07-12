""" Implementation of a Calibration Plan which is bound to
    - contain a name and plan
    - execute plan
"""

import pandas as pd
import sys
from time import process_time_ns, time
from typing import List

from ..utils import flatten_dicts
from ..data.data_transformer import DataTransformer
from ..data.data_object_manager import DataObjectManager
from ..calibration.calibration_engine import CalibrationEngine
import logging
from multiprocessing import Process, Manager, Queue
from time import sleep
import pickle
import numbers


class CalibrationPlan:
    def __init__(
        self, name: str, plan: dict, yaml_name: str, verbosity_level: int
    ) -> None:
        """Constructor of this class

        Args:
            name (str): Plan name
            plan (dict): Plan instructions
            verbosity_level: Integer indicating the level of verbosity
        """
        self._name = name
        self._plan = plan
        self._yaml_name = yaml_name
        self._metadata = pd.DataFrame()
        self._verbosity_level = verbosity_level
        self._state = {
            "calibration_models": [],
        }

    @property
    def name(self) -> str:
        """Name property which is set to read-only

        Returns:
            str: Plan name
        """
        return self._name

    @property
    def plan(self) -> dict:
        """plan property which is set to read-only

        Returns:
            dict: Plan instructions
        """
        return self._plan

    @property
    def yaml_name(self) -> str:
        """yaml plan name property which is set to read-only

        Returns:
            str: yml plan name
        """
        return self._yaml_name

    @property
    def metadata(self) -> pd.DataFrame:
        """metadata property which is set to read-only

        Returns:
            pd.DataFrame: DataFrame with scores
        """
        return self._metadata

    @property
    def verbosity_level(self) -> int:
        """verbosity_level property which is set to read-only

        Returns:
            int: verbosity_level
        """
        return self._verbosity_level

    @verbosity_level.setter
    def verbosity_level(self, value):
        """sets verbosity level

        Args:
            value (int): new vebosity level
        """
        self._verbosity_level = value

    def intitialize_metadata(self):
        """Constructs an empty dataframe with prefilled columns"""
        self._metadata = pd.DataFrame(
            columns=[
                "Model Name",
                "Model Method",
                "Score",
                "Difference Function",
                "Dataframe",
                "Duration",
            ]
        )

    def sort_metadata(self):
        return NotImplementedError

    def get_state(self) -> str:
        """Get state property

        Returns:
            str: Plan state
        """
        return self._state

    def run(self, multiprocessing: bool = True):
        """Method that runs the plan instructions

        Args:
            multiprocessing (bool, optional): Whether or not to run the plan instructions with multiprocessing. Defaults to True.
        """
        # Initialize metadata

        mgr = Manager()
        ns = mgr.Namespace()
        ns.shared_data = {}

        self.intitialize_metadata()
        data = self.plan["data"]
        for d in data:
            d = list(d.values())[0]
            data_manager = DataObjectManager()
            cols_incl = []
            cols_excl = []
            if d["columns_included"]:
                cols_incl = d["columns_included"]
            if d["columns_excluded"]:
                cols_excl = d["columns_excluded"]
            data_manager.read_from_csv(
                file=d["file"], name=d["name"], cols_incl=cols_incl, cols_excl=cols_excl
            )

        # Transformations
        if "transformations" in self.plan.keys():
            transformations = self.plan["transformations"]
            for t in transformations:
                t = list(t.values())[0]
                steps = t["steps"]
                df = data_manager.get_dataframe(name=t["in"])
                for s in steps:
                    s = list(s.values())[0]
                    dtf = DataTransformer().get_datatransform(name=s["method"])
                    df = dtf.transform(
                        df_in=df,
                        debug=True if self.verbosity_level > 1 else False,
                        **flatten_dicts(s["args"]) if s["args"] else {},
                    )
                data_manager.add_dataframe(name=t["out"], df=df)

        # Create processes for all models for each calibration
        calibration_global_parameters_models = dict()
        calibration_global_parameters_reports = dict()
        calibration_global_parameters_difference_functions = dict()
        if "calibrations_global_parameters" in self.plan.keys():
            calibration_global_parameters = self.plan["calibrations_global_parameters"]
            if "models" in calibration_global_parameters.keys():
                calibration_global_parameters_models = calibration_global_parameters[
                    "models"
                ]
            if "difference_functions" in calibration_global_parameters.keys():
                calibration_global_parameters_difference_functions = (
                    calibration_global_parameters["difference_functions"]
                )
            if "reports" in calibration_global_parameters.keys():
                calibration_global_parameters_reports = (
                    calibration_global_parameters["reports"]
                    if calibration_global_parameters["reports"]
                    else {}
                )

        if "calibrations_models" in self.plan.keys():
            calibrations_models = self.plan["calibrations_models"]
            self.process_models = {}
            self.process_models_plugin_scope = {}
            self.process_id = 0
            self.processes = {}
            self.calibration_models_objects = []
            self.calibration_models_done = []
            for cm in calibrations_models:
                cm = list(cm.values())[0]
                model = cm["model"]
                df = data_manager.get_dataframe(name=cm["in"])

                # Calibration Difference functions
                for diff in model["difference_functions"]:
                    calibration_model = CalibrationEngine().get_calibration_model(
                        name=model["method"],
                    )
                    diff = list(diff.values())[0]
                    diff = flatten_dicts(diff)

                    diff_func_obj = CalibrationEngine().get_calibration_diff_func(
                        name=diff["method"]
                        # debug=True if self.verbosity_level > 1 else False,
                    )
                    calibration_model.diff_func_object = diff_func_obj

                    # Model parameters
                    model_parameters = model["params"] if model["params"] else {}
                    merged_model_parameters = {}
                    for d in calibration_global_parameters_models:
                        for k, v in d.items():
                            #add for datamanager to get also data from transformation
                            if isinstance(v, str) and (v in data_manager.__dict__["_data"]):
                                merged_model_parameters[k] = data_manager.get_dataframe(name=v)
                            else:
                                merged_model_parameters[k] = v
                    for d in model_parameters:
                        for k, v in d.items():
                            if isinstance(v, str) and (v in data_manager.__dict__["_data"]):
                                merged_model_parameters[k] = data_manager.get_dataframe(name=v)
                            else:
                                merged_model_parameters[k] = v

                    # Diff func parameters
                    diff_func_parameters = dict()
                    if "params" in diff.keys():
                        diff_func_parameters = diff["params"]
                    merged_diff_func_parameters = {}

                    for d in calibration_global_parameters_difference_functions:
                        for k, v in d.items():
                            merged_diff_func_parameters[k] = v
                    for d in diff_func_parameters:
                        for k, v in d.items():
                            merged_diff_func_parameters[k] = v

                    # Report parameters
                    merged_report_parameters = {}
                    if "report" in model:
                        report = model["report"]
                        report = flatten_dicts(report)
                        report_parameters = report["params"] if report["params"] else {}

                        for d in calibration_global_parameters_reports:
                            for k, v in d.items():
                                merged_report_parameters[k] = v
                        for d in report_parameters:
                            for k, v in d.items():
                                merged_report_parameters[k] = v
                    info = {
                        "cm_name": cm["name"],
                        "model_method": model["method"],
                        "diff_func_name": diff["name"],
                        "dataframe_in": cm["in"],
                        "parameters": merged_model_parameters,
                        "diff_func_parameters": merged_diff_func_parameters,
                        "report_parameters": merged_report_parameters,
                    }
                    calibration_model.append_info(info)

                    data = {}
                    data["df_in"] = df
                    data["debug"] = True if self.verbosity_level > 1 else False
                    data["diff_func_object"] = diff_func_obj
                    data["diff_func_parameters"] = merged_diff_func_parameters
                    data["kwargs"] = merged_model_parameters

                    if multiprocessing:
                        # Do not compact this
                        shared_data = ns.shared_data
                        shared_data[self.process_id] = data
                        ns.shared_data = shared_data

                        self.process_models_plugin_scope[self.process_id] = mgr.dict()
                        psp = self.process_models_plugin_scope[self.process_id]

                        p = Process(
                            target=calibration_model.run,
                            args=(ns, self.process_id, psp),
                        )

                        self.process_models[self.process_id] = calibration_model
                        self.processes[self.process_id] = p
                        self.process_id = self.process_id + 1
                    else:
                        start_time = time()
                        debug = True if self.verbosity_level > 1 else False
                        calibration_model.fit(
                            df_in=df,
                            diff_func_object=diff_func_obj,
                            diff_func_parameters=merged_diff_func_parameters,
                            debug=debug,
                            **merged_model_parameters,
                        )
                        end_time = time()
                        self.calibration_models_done.append(calibration_model)
                        info = {"running_time": end_time - start_time}
                        calibration_model.append_info(info)

            if multiprocessing:
                for p in self.processes.values():
                    p.start()

                # todo: wait
                for p in self.processes.values():
                    p.join()

                # Update model self parameters
                # We use this to maintain a consistent 'self' in the plugins
                for pid in self.process_models_plugin_scope:
                    plugin_scope = self.process_models_plugin_scope[pid]
                    calibration_model = self.process_models[pid]
                    calibration_model.pluginscope = plugin_scope["data"]
                    running_time = calibration_model.pluginscope["__runningtime__"]
                    info = {"running_time": running_time}
                    calibration_model.append_info(info)

                self.calibration_models_done = self.process_models.values()

            # Post process models
            for calibration_model in self.calibration_models_done:
                model_info = calibration_model.get_info()
                # Add score to metadata
                duration = (
                    model_info["running_time"]
                    # calibration_model.pluginscope["__runningtime__"]
                    # if multiprocessing
                    # else end_time - start_time
                )
                min, sec = divmod(duration, 60)
                data = pd.Series(
                    {
                        "Model Name": model_info["cm_name"],
                        "Model Method": model_info["model_method"],
                        "Score": round(calibration_model.get_score(), 2),
                        "Difference Function": model_info["diff_func_name"],
                        "Dataframe": model_info["dataframe_in"],
                        "Duration": "{:.3f} sec".format(duration),
                        "Solution Params": calibration_model.get_summary()[0],
                    }
                )
                #changed this because of update from pandas
                self.metadata.loc[self.metadata.index.max() + 1] = data
                self._metadata = self.metadata.reset_index(drop=True)

                if self.verbosity_level > 1:
                    print(calibration_model.get_summary())

                model_report = calibration_model.get_report()
                self._state["calibration_models"] += [calibration_model]
