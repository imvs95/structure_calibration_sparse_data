""" Implementation of the Calibration Engine (Singleton) which is bound to
    - parse yaml and plugins
    - interact with a CalibrationModel and difference function(s)
"""
import logging
from typing import List
from .calibration_model import CalibrationModel
from .calibration_difference_function import CalibrationDifferenceFunction
from ..config_manager import ConfigurationManager
from ..plugins.plugin_helper_methods import *
from ..utils import Singleton
import yaml
import os
import sys
import logging

from celibration.plugins import plugin_helper_methods


class CalibrationEngine(metaclass=Singleton):
    def initialize(self, *args, **kwargs):
        """Overrides default constructor behaviour by:
        - Manually setting attributes
        - Import calibration related plugins
        """
        self._plugin_info_list = []
        self._plugins = {}
        self._plugin_info_list_diff_funcs = []
        self._plugins_diff_funcs = {}
        plugins_folder = ConfigurationManager().get_config_value("plugins_folder")
        plugins_folder = [plugins_folder] + kwargs["custom_plugin_dir"]
        self.import_plugins(plugins_folder=plugins_folder)

    def __del__(self):
        pass

    def import_plugins(self, plugins_folder: str):
        """Traverse a given directory (plugins_folder) and parse each subsequent plugin yaml file

        Args:
            plugins_folder (str): Location of the plugins folder
        """
        self._plugin_info_list = []
        self._plugins = {}
        self._plugin_info_list_diff_funcs = []
        self._plugins_diff_funcs = {}

        logging.info("Calibration model plugins:")
        for folder in plugins_folder:
            for dirpath, dirs, files in os.walk(folder):
                # Read yaml files
                for filename in files:
                    filepath = os.path.join(dirpath, filename)
                    root, ext = os.path.splitext(filepath)
                    if ext == ".yaml":
                        self.parse_yaml(dirpath, filename)

    def get_plugin_info(self) -> List[dict]:
        """Gives the current parsed plugin information as a list

        Returns:
            List[dict]: A list of dicts where each dict contains a plugin name (key) and corresponding implementation script name (value)
        """
        return self._plugin_info_list + self._plugin_info_list_diff_funcs

    def get_calibration_model(self, name: str) -> CalibrationModel:
        """Gives a CalibrationModel instance if the provided name occurs in the plugins

        Args:
            name (str): Name of the plugin

        Raises:
            KeyError: The name provided does not exist in the plugins list

        Returns:
            CalibrationModel: Instance of a CalibrationModel
        """
        try:
            if name not in self._plugins.keys():
                raise KeyError
            else:
                plugin_info = [i for i in self._plugin_info_list if i["name"] == name][
                    0
                ]
                return self._plugins[name].create_calibration_model(
                    self._plugins[name],
                    info=plugin_info,
                )
        except KeyError as err:
            raise KeyError(
                f"Please provide a valid name {list(self._plugins.keys())}, {err}"
            )

    def get_calibration_diff_func(
        self,
        name: str,
    ) -> CalibrationDifferenceFunction:
        """Gives a CalibrationModelDifferenceFunction instance if the provided name occurs in the CalibrationModelDifferenceFunction plugins

        Args:
            name (str): Name of the CalibrationModelDifferenceFunction plugin

        Raises:
            KeyError: The name provided does not exist in the plugins list

        Returns:
            CalibrationDifferenceFunction: Instance of a CalibrationModelDifferenceFunction
        """
        try:
            if name not in self._plugins_diff_funcs.keys():
                raise KeyError
            else:
                plugin_info = [
                    i for i in self._plugin_info_list_diff_funcs if i["name"] == name
                ][0]
                return self._plugins_diff_funcs[
                    name
                ].create_calibration_difference_function(
                    self._plugins_diff_funcs[name], info=plugin_info
                )
        except KeyError as err:
            raise KeyError(
                f"{name} does not occur in {list(self._plugins_diff_funcs.keys())}, {err}"
            )

    def parse_yaml(self, dirpath: str, filename: str):
        """Parse calibration related (yaml) plugins from a given directory path (dirpath) and filename. The plugin infomration is then stored for later usage.
            - CalibrationModel plugins
            - CalibrationDifferenceFunctions plugins

        Args:
            dirpath (str): Full path to the current directory
            filename (str): Name of the plugin yaml
        """
        filepath = os.path.join(dirpath, filename)
        plugin_info = load_plugin_yaml(filepath)

        # CalibrationModel plugins
        if plugin_info[plugin_type_key()] == "calibrationmodel":
            try:
                plugin_factory_class = import_plugin_module(
                    dirpath,
                    "CalibrationModelFactory",
                )
            except Exception as e:
                logging.error(
                    f"Couldn't import CalibrationModelFactory {plugin_info[plugin_name_key()]}"
                )
                logging.error(e)
                return

            self._plugin_info_list.append(plugin_info)
            self._plugins[plugin_info[plugin_name_key()]] = plugin_factory_class
            print(
                f"\t{plugin_info[plugin_name_key()]} {plugin_info[plugin_version_key()]}"
            )

        # CalibrationDifferenceFunctions plugins
        if plugin_info[plugin_type_key()] == "calibrationdifferencefunction":
            try:
                plugin_factory_class = import_plugin_module(
                    dirpath,
                    "CalibrationDifferenceFunctionFactory",
                )
            except Exception as e:
                logging.error(
                    f"Couldn't import CalibrationDifferenceFunction {plugin_info[plugin_name_key()]}"
                )
                logging.error(e)
                return

            self._plugin_info_list_diff_funcs.append(plugin_info)
            self._plugins_diff_funcs[
                plugin_info[plugin_name_key()]
            ] = plugin_factory_class
            print(
                f"\t{plugin_info[plugin_name_key()]} {plugin_info[plugin_version_key()]}"
            )
