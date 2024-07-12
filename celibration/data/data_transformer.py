""" Implementation of a Data Transformer Object which is bound to
    - parse yaml and plugins
    - interact with the DataTransform Objects
"""

from typing import List

from ..utils import Singleton
from .data_object_manager import DataObjectManager
from .data_transform import DataTransform
from ..config_manager import ConfigurationManager
from ..plugins.plugin_helper_methods import *
import yaml
import os
import sys
import logging


class DataTransformer(metaclass=Singleton):
    def initialize(self, *args, **kwargs):
        """Overrides default constructor behaviour by:
        - Manually setting attributes
        - Import datatransform related plugins
        """
        self._plugin_info_list = []
        self._plugins = {}
        plugins_folder = ConfigurationManager().get_config_value("plugins_folder")
        # print(plugins_folder, kwargs["custom_plugin_dir"])
        plugins_folder = [plugins_folder] + kwargs["custom_plugin_dir"]
        self.import_plugins(plugins_folder=plugins_folder)

    def __del__(self):
        pass

    # Make this generic later on?
    def import_plugins(self, plugins_folder: list):
        """Traverse a given directory (plugins_folder) and parse each subsequent plugin yaml file

        Args:
            plugins_folder (str): Location of the plugins folder
        """
        self._plugin_info_list = []
        self._plugins = {}
        logging.info("Found transform plugins:")
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
        return self._plugin_info_list

    def get_datatransform(self, name: str) -> DataTransform:
        """Gives a DataTransform instance if the provided name occurs in the plugins

        Args:
            name (str): Name of the plugin

        Raises:
            KeyError: KeyError: The name provided does not exist in the plugins list

        Returns:
            DataTransform: Instance of a DataTransform
        """
        try:
            if name not in self._plugins.keys():
                raise KeyError
            else:
                plugin_info = [i for i in self._plugin_info_list if i["name"] == name][
                    0
                ]
                return self._plugins[name].create_transform(
                    self._plugins[name], info=plugin_info
                )
        except KeyError as err:
            raise KeyError(
                f"Please provide a valid name {list(self._plugins.keys())}, {err}"
            )

    def parse_yaml(self, dirpath: str, filename: str):
        """Parse datatransform related (yaml) plugins from a given directory path (dirpath) and filename. The plugin infomration is then stored for later usage.
            - DataTransform plugins

        Args:
            dirpath (str): Full path to the current directory
            filename (str): Name of the plugin yaml
        """
        filepath = os.path.join(dirpath, filename)
        plugin_info = load_plugin_yaml(filepath)

        # DataTransform plugins
        if plugin_info[plugin_type_key()] == "datatransform":
            try:
                plugin_factory_class = import_plugin_module(
                    dirpath,
                    "DataTransformFactory",
                )
            except ModuleNotFoundError:
                logging.error(
                    f"Couldn't import DataTransformFactory {plugin_info[plugin_name_key()]}"
                )
                return

            self._plugin_info_list.append(plugin_info)
            self._plugins[plugin_info[plugin_name_key()]] = plugin_factory_class
            print(
                "\t", plugin_info[plugin_name_key()], plugin_info[plugin_version_key()]
            )
