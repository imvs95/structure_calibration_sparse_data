import pandas as pd
from celibration import DataTransform, DataTransformFactory

import logging


class PluginFactory(DataTransformFactory):
    def create_transform(self, info=dict) -> DataTransform:
        return DataPreparationTransform(info=info)


class DataPreparationTransform(DataTransform):
    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs) -> pd.DataFrame:
        """This functions prepares the data such that it has the format which
        is applicable to the samplers.

        Args:
            df_in (dataframe): Dataset

        Returns:
            df_out: Prepared dataset
        """
        df = df_in.set_index("Time")
        df_out = df.groupby(["Time"]).mean().iloc[:, 1:-1] #with replication - use mean?
        if debug:
            logging.info("Data Preparation")
        return df_out
