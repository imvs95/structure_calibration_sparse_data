import pandas as pd
import numpy as np
from celibration import DataTransform, DataTransformFactory

import logging


class PluginFactory(DataTransformFactory):
    def create_transform(self, info=dict) -> DataTransform:
        return PostProcessingTransform(info=info)


class PostProcessingTransform(DataTransform):
    def calculate_average_interval_time(self, column):
        shifted_column = column.shift(1)
        # if previous item on list is smaller than current
        bool_comparison = shifted_column < column
        # get times
        index_restock = bool_comparison.index[bool_comparison == True]

        if len(index_restock) == 0:
            avg_interval_time = 0

        else:
            # determine interval times
            interval_times = [index_restock[0] - 0] + [index_restock[n] - index_restock[n - 1] for n in
                                                       range(1, len(index_restock))]

            # calculate average
            avg_interval_time = np.mean(interval_times)

        return avg_interval_time

    def transform(self, df_in: pd.DataFrame, debug: bool, **kwargs) -> pd.DataFrame:
        """This functions prepares the data such that it aggregates the statistics of time series,
        and can be used for define the quality of fit.

        Args:
            df_in (dataframe): Dataset

        Returns:
            df_out: Prepared dataset
        """

        df_out = pd.DataFrame(index=["mean", "std", "p5", "p95", "avg_interval_t"], columns=df_in.columns)
        df_out.loc["mean"] = [np.mean(df_in[col]) for col in df_in.columns]
        df_out.loc["std"] = [np.std(df_in[col]) for col in df_in.columns]
        df_out.loc["p5"] = [np.nanpercentile(df_in[col], 5) for col in df_in.columns]
        df_out.loc["p95"] = [np.nanpercentile(df_in[col], 95) for col in df_in.columns]
        df_out.loc["avg_interval_t"] = [self.calculate_average_interval_time(df_in[col]) for col in df_in.columns]

        return df_out
