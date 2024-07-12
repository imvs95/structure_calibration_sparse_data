import pandas as pd

from celibration import DataTransform, DataTransformFactory

import logging
import pickle

class PluginFactory(DataTransformFactory):
    def create_transform(self, info=dict) -> DataTransform:
        return LoadPickleTransform(info=info)


class LoadPickleTransform(DataTransform):
    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs):
        """This functions prepares the data such that it has the format which
        is applicable to the samplers.

        Args:
            df_in (dataframe): Dataset

        Returns:
            df_out: Prepared dataset
        """
        with open(kwargs["file_pkl"], "rb") as f:
            pickle_data = pickle.load(f)
        logging.info("Pickle loaded " + kwargs["file_pkl"])

        return pickle_data
