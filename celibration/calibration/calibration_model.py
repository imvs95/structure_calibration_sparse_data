""" Abstract definition of the building blocks of a CalibrationModel object
"""

from typing import Callable, Tuple
import pandas as pd
from abc import ABC, abstractmethod
from .calibration_difference_function import CalibrationDifferenceFunction
from ..utils import sharedmemory
from multiprocessing import Process, Manager
from multiprocessing.managers import Namespace
from time import sleep, time
from ..utils import flatten_dicts


class CalibrationModel(ABC):
    def __init__(self, info: dict):
        self._info = info
        self._diff_func_object = None
        self.p = None
        self.pluginscope = {}

    def get_info(self) -> dict:
        # Changed this to use less memory with many graphs
        model_info = self._info.copy()
        for k, v in model_info["parameters"].items():
            if isinstance(v, dict):
                model_info["parameters"][k] = {"max_keys": max(list(v.keys()))}
            elif isinstance(v, pd.DataFrame):
                model_info["parameters"][k] = {"len_df": len(v), "columns_df": list(v.columns)}
            else:
                continue
        return model_info

    def append_info(self, info: dict):
        """Add key-value pairs to an existing info class attribute

        Args:
            info (dict): Information stored in key-value pairs
        """

        self._info = {**self._info, **info}

    @property
    def diff_func_object(self) -> int:
        """Difference function property which is set to read-only

        Returns:
            int: verbosity_level
        """
        return self._diff_func_object

    @diff_func_object.setter
    def diff_func_object(self, value):
        """sets Difference function

        Args:
            value (int): new vebosity level
        """
        self._diff_func_object = value

    def run(self, ns: Namespace, id: dict, pluginscope: dict):
        """Run function of a calibration model

        Args:
            ns (Namespace): Multiprocess Manager's Namespace
            id (dict): Process id
            pluginscope (dict): Captured scope of the current class
        """
        shared_data = ns.shared_data
        data = shared_data[id]
        df = data["df_in"]
        diff_func_object = data["diff_func_object"]
        diff_func_parameters = data["diff_func_parameters"]
        debug = data["debug"]
        kwargs = data["kwargs"]

        start = time()
        self.fit(
            df_in=df,
            diff_func_object=diff_func_object,
            diff_func_parameters=diff_func_parameters,
            debug=debug,
            **kwargs
        )
        end = time()

        self.pluginscope["__runningtime__"] = end - start
        pluginscope["data"] = self.pluginscope

    # @sharedmemory
    @abstractmethod
    def fit(
        self,
        df_in: pd.DataFrame,
        diff_func_object: CalibrationDifferenceFunction,
        diff_func_parameters: dict,
        debug: bool,
        **kwargs,
    ):
        """Abstract method that defines a generic fit function which accepts:
            - dataframe in (required)
            - difference function (required)
            - additional model specific params (kwargs)

        Args:
            df_in (pd.DataFrame): Dataframe which the model will use for training/optimization
            diff_func_object (CalibrationDifferenceFunction): Object that holds a difference function
            debug (bool): Boolean indicating if logging verbosity should be applied
        """
        return NotImplementedError

    @abstractmethod
    def get_summary(self) -> Tuple[dict, pd.DataFrame]:
        return NotImplementedError

    @abstractmethod
    def get_score(self) -> float:
        return NotImplementedError

    @abstractmethod
    def get_report(self, **kwargs):
        return NotImplementedError
