""" Abstract definition of the building blocks of a DataTransform object
"""

import pandas as pd
from abc import ABC, abstractmethod


class DataTransform(ABC):
    def __init__(self, info):
        self._info = info

    def get_info(self):
        return self._info

    @abstractmethod
    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs) -> pd.DataFrame:
        """Abstract method that defines a calculation given a difference function and two types of input

        Args:
            df_in (pd.DataFrame): DataFrame
            debug (bool): Boolean indicating if logging verbosity should be applied
            **kwargs (Any): additional arguments needed for the chosen transform plugin
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        return NotImplementedError
