""" Implementation of the Data Object Manager (Singleton) which is bound to
    - read data from csv
    - create, delete and get DataFrames
    - interact with DataObjects
"""

import pandas as pd
from ..utils import Singleton


class DataObjectManager(metaclass=Singleton):
    def initialize(self, *args, **kwargs):
        """Overrides default constructor behaviour by:
        - Manually setting attributes
        """
        self._data = {}

    def __del__(self):
        pass

    def read_from_csv(self, file: str, name: str, cols_incl: list, cols_excl: list):
        """Reads CSV data from a provided path and parameters

        Args:
            file (str): Full path of the file including the extension (.csv)
            name (str): Name of the dataframe
            cols_incl (list): List of column names to be included (empty means using all columns)
            cols_excl (list): List of columns names to be excluded (empty means using all columnss)

        Raises:
            Exception: Something went wrong when parsing the csv file. Either the path is not correct, file may be corrupt or additional parameters need to be set in order to parse the content succesfully.
        """
        try:
            if cols_incl:
                df = pd.read_csv(file, sep=",", usecols=cols_incl)
            else:
                df = pd.read_csv(file, sep=",")

            if cols_excl:
                df.drop(cols_excl, inplace=True, axis=1)

            self.add_dataframe(name=name, df=df)
        except Exception as err:
            raise KeyError(f"Something went wrong, see {err}")

    def get_names(self) -> list:
        """Gives the names of all data objects as a list

        Returns:
            list: A list of data object names
        """
        return list(self._data.keys())

    def get_dataframe(self, name: str) -> pd.DataFrame:
        """Gives a DataFrame from a provided name

        Args:
            name (str): Name of the DataObject in which the requested DataFrame is stored in

        Raises:
            KeyError: The provided name does not occur in the data object list, are you sure that you added this dataframe?

        Returns:
            pd.DataFrame: DataFrame of the requested name
        """
        try:
            if name not in self._data.keys():
                raise KeyError
            else:
                return self._data[name]
        except KeyError as err:
            raise KeyError(
                f"Please provide a valid name {list(self._data.keys())}, {err}"
            )

    def add_dataframe(
        self,
        name: str,
        df: pd.DataFrame,
    ):
        """Adds a dataframe to the current storage

        Args:
            name (str): Name of the DataFrame
            df (pd.DataFrame): DataFrame object

        Raises:
            KeyError: Duplicate keys are not allowed, please provide a new name as the current one already exists
        """
        try:
            if name in self._data.keys():
                raise KeyError
            else:
                self._data.update({name: df})
        except KeyError as err:
            raise KeyError(
                f"Not allowed to add duplicate keys {list(self._data.keys())}, {err}"
            )

    def remove_dataframe(self, name: str) -> bool:
        """Deletes a DataFrame by a provided name

        Args:
            name (str): Name of the requested DataFrame to be deleted

        Returns:
            bool: True if name occurs in the data storage and is successfully removed and False otherwise
        """
        if name in self._data.keys():
            self._data = {k: v for k, v in self._data.items() if k != name}
            return True
        else:
            return False
