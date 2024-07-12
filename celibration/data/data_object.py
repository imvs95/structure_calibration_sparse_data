""" Implementation of a Data Object which is bound to
    - contain a name, dataframe, objectives (and possibly one or more parents)
    - import and export methods
"""

import pandas as pd
from typing import List


class DataObject:
    def __init__(
        self, df: pd.DataFrame, name: str, objectives: List[str], parent: List[str]
    ) -> None:
        """Constructor of this class

        Args:
            df (pd.DataFrame): DataFrame
            name (str): Name of the DataObject (unique)
            objectives (List[str]): List of column name(s) which act as objectives
            parent (List[str]): List of this DataObject's parents
        """
        self._df = df
        self._name = name
        self._objectives = objectives
        self._parent = parent

    @property
    def df(self) -> pd.DataFrame:
        """df property which is set to read-only

        Returns:
            pd.DataFrame: Return dataframe instance of this Data Object
        """
        return self._df

    @property
    def name(self) -> str:
        """Name property which is set to read-only

        Returns:
            str: Return the name of this Data Object
        """
        return self._name

    @property
    def objectives(self) -> List[str]:
        """Objectives property which is set to read-only

        Returns:
            List[str]: Return the objective(s) of this Data Object
        """
        return self._objectives

    @property
    def parent(self) -> List[str]:
        """Parent property which is set to read-only

        Returns:
            List[str]: Return parent(s) of this Data Object
        """
        return self._parent
