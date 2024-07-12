import pandas as pd
from celibration import DataTransform, DataTransformFactory

import numpy as np
import logging


class PluginFactory(DataTransformFactory):
    def create_transform(self, info=dict) -> DataTransform:
        return BiasSamplerDataTransform(info=info)


class BiasSamplerDataTransform(DataTransform):
    def sample_bias(self, dataframe, bias_percentage):
        """This functions draws an user-defined biased sample from the dataset and combines
        this with the normal dataset. The distribution used for this biased sample set is a
        LogNormal distribution.

        Args:
            dataframe (dataframe): Dataset
            bias_percentage (float): Percentage of bias to be implemented

        Returns:
            dataframe: Dataset with bias
        """

        lognormal = np.random.lognormal(size=len(dataframe))
        rows_to_sample = round(len(dataframe) * bias_percentage)
        sample = dataframe.sample(rows_to_sample, weights=lognormal, replace=True)
        no_sample = dataframe.sample(len(dataframe) - rows_to_sample, replace=False)
        combined_sample = (
            pd.concat([sample, no_sample]).sort_index().reset_index(drop=True)
        )
        return combined_sample

    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs) -> pd.DataFrame:
        """This functions transforms the dataframe with bias of an user-defined percentage.
        A seed can be set by the user to control the randomness.

        Args:
            df_in (dataframe): Dataset
            percentage (float): Percentage of bias to be implemented

        Returns:
            df_out: Dataset with bias
        """
        if "seed" in kwargs:
            np.random.seed(kwargs["seed"])

        if "columns_to_transform" in kwargs:
            df_in_transform = df_in[kwargs["columns_to_transform"]]
            df_out_transform = self.sample_bias(dataframe=df_in_transform, bias_percentage=kwargs["percentage"])
            df_out = df_in.copy()
            df_out[kwargs["columns_to_transform"]] = df_out_transform[kwargs["columns_to_transform"]]

        else:
            df_out = self.sample_bias(dataframe=df_in, bias_percentage=kwargs["percentage"])

        logging.info(
                "Done: Bias sampler percentage {0:.2f} of the values".format(
                    kwargs["percentage"]
                )
            )

        return df_out
