""" Implementation of a Configuration Manager which is bound to
    - parse a configuration yaml
"""

from typing import Any
import yaml
from .utils import Singleton
from pathlib import Path


class ConfigurationManager(metaclass=Singleton):
    def initialize(self, *args, **kwargs):
        """Overrides default constructor behaviour by:
        - Manually setting attributes
        - Import datatransform related plugins
        """
        self._config = dict()
        self.read_from_yaml(kwargs["config"])

    def __del__(self):
        pass

    def read_from_yaml(self, filename: str):
        """Reads configuration yaml file from a given filepath

        Args:
            filename (str): Full path to the file (including filename)

        Raises:
            Exception: Something went wrong reading the config file
        """
        try:
            with open(filename, "r") as f:
                config = yaml.full_load(f)

                # Overwrite plugin_folder variable with correct path for prod
                if config["prod"]:
                    try:
                        import celibration

                        p = Path(celibration.__file__).parents[0] / "plugins"
                        config["plugins_folder"] = str(p.resolve())
                    except ModuleNotFoundError as err:
                        raise f"Something went wrong, see {err}"
                else:
                    config["plugins_folder"] = str(
                        Path(config["plugins_folder"]).resolve()
                    )
                self.store_config(config=config)
        except Exception as err:
            raise KeyError(f"Something went wrong reading the config file, see {err}")

    def store_config(self, config: dict):
        """Store the configuration content

        Args:
            config (dict): Configuration content
        """
        self._config = config

    def get_config(self) -> dict:
        """Gives the current configuration settings

        Returns:
            dict: Configuration settings
        """
        return self._config

    def get_config_value(self, key: str) -> Any:
        """Gives the value of a configuration setting from a key

        Args:
            key (str): The name (key) of the requested configuration setting

        Raises:
            KeyError: Configuration key not found

        Returns:
            Any: Value of the requested configuration setting
        """
        try:
            return self._config[key]
        except KeyError as err:
            raise KeyError(f"Configuration key not found, {err}")
