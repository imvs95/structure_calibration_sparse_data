""" Implementation of the Calibration Plan Manager (Singleton) which is bound to
    - parse yaml
    - create plans
    - interact with a CalibrationModel and difference function(s)
"""

from typing import List
import pandas as pd
import yaml

from celibration.data.data_object_manager import DataObjectManager
from .calibration_plan import CalibrationPlan
from ..config_manager import ConfigurationManager
from ..data.data_transformer import DataTransformer
from .calibration_engine import CalibrationEngine
from ..plugins.plugin_helper_methods import *
from ..utils import EventHook, Singleton
from pathlib import Path
import coloredlogs, logging
from time import time

from celibration import config_manager

import warnings

warnings.filterwarnings("ignore")

coloredlogs.install(level="INFO")


class CalibrationPlanManager(metaclass=Singleton):
    def initialize(self, *args, **kwargs):
        """Overrides default constructor behaviour by:
        - Manually setting attributes
        - Start DataTransformer worker (Singleton)
        - Start CalibrationEngine worker (Singleton)
        """
        self._plans = []
        self._metadata = {}
        self._verbosity_level = 0
        self.onPlanUpdated = EventHook()
        if "custom_plugin_dir" not in kwargs:
            kwargs["custom_plugin_dir"] = []

        self.configuration_manager = ConfigurationManager(config=kwargs["config"])
        # Add sys path for plugin references
        plugins_folder = self.configuration_manager.get_config_value("plugins_folder")
        plugins_folder = [plugins_folder] + kwargs["custom_plugin_dir"]
        for folder in plugins_folder:
            dirs = [
                name
                for name in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, name))
            ]
            for subdir in dirs:
                sys.path.append(os.path.join(os.path.abspath(folder), subdir))

        self.data_object_manager = DataObjectManager()
        self.data_transformer = DataTransformer(
            custom_plugin_dir=kwargs["custom_plugin_dir"]
        )
        self.calibration_engine = CalibrationEngine(
            custom_plugin_dir=kwargs["custom_plugin_dir"]
        )

    def __del__(self):
        pass

    def set_verbosity(self, level: int):
        """Sets the level of verbosity

        Args:
            level (int): Integer indicating the verbosity level
        """
        self._verbosity_level = level
        if self._verbosity_level == 0:
            logging.basicConfig(level=logging.WARN)
        elif self._verbosity_level == 1:
            logging.basicConfig(level=logging.INFO)
        else:
            logging.basicConfig(level=logging.DEBUG)

    def get_plans(self) -> List[dict]:
        """Give the current plans

        Returns:
            List[dict]: A list of dicts where each dict contains a plan name (key) and corresponding content/instructions (value)
        """
        return self._plans

    def read_plan_from_yaml(self, filename: str):
        """Reads a yaml file and redirect the content as a dict to a parsing function

        Args:
            filename (str): Name of the yaml file that contains one or more plans

        Raises:
            Exception: Error parsing the yaml file
        """
        try:
            with open(filename, "r") as f:
                plan = yaml.full_load(f)
                self.parse(plans=plan, filename=filename)
        except Exception as err:
            raise KeyError(f"Something went wrong, see {err}")

    def parse(self, plans: dict, filename: str):
        """Extract plans

        Args:
            plans (dict): Nested dict structure that contains at least one plan
            filename (str): Name of the yaml file that contains one or more plans
        """
        plan_yaml_path = PurePath(filename)
        yaml_name = str(plan_yaml_path.name).split(".yml")[0]
        for p in plans["plans"]:
            p = list(p.values())[0]
            self._plans.append(
                CalibrationPlan(
                    name=p["name"],
                    plan=p,
                    yaml_name=yaml_name,
                    verbosity_level=self._verbosity_level,
                )
            )

    def on_plan_state_updated(self, plan: CalibrationPlan):
        """Method that triggers an event at the time a plan needs to be updated

        Args:
            plan (CalibrationPlan): A calibration plan object
        """
        self.onPlanUpdated.fire(plan=plan)

    def run(self, multiprocessing: bool = False):
        """Runs each stored plan

        Args:
            multiprocessing (bool, optional): Whether or not to run the plan with multiprocessing enabled. Defaults to False.
        """
        logging.info(
            f"******************** Started running plans (multiprocessing: {multiprocessing}) ********************"
        )
        start_time = time()
        for p in self._plans:
            p.run(multiprocessing=multiprocessing)
            self.on_plan_state_updated(plan=p)
            metadata = p.metadata
            self._metadata[p.name] = metadata
            model_comparison = (
                p.plan["model_comparison"]
                if "model_comparison" in p.plan.keys()
                else None
            )
            if model_comparison:
                if model_comparison["sorting_order_priority"]:
                    columns = model_comparison["sorting_order_priority"]
                    sort_direction = True
                    sort_direction = (
                        sort_direction
                        if model_comparison["sorting_direction"]
                        and model_comparison["sorting_direction"] == "ASC"
                        else False
                    )
                    metadata.sort_values(
                        by=columns,
                        ascending=sort_direction,
                        inplace=True,
                    )
                    metadata = metadata.reset_index(drop=True)

            logging.info("Calibration summary:")
            print(p.name)
            print(metadata)
        end_time = time()
        min, sec = divmod(end_time - start_time, 60)
        log_text = "******************** Execution of plans took {:.0f} minutes and {:.3f} seconds ********************".format(
            min, sec
        )
        logging.info(log_text)
