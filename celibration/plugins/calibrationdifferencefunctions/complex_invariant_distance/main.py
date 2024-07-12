import pandas as pd
import numpy as np

from typing import Any
from celibration import (
    CalibrationDifferenceFunction,
    CalibrationDifferenceFunctionFactory,
)

from ema_workbench.analysis.clusterer import calculate_cid


class PluginFactory(CalibrationDifferenceFunctionFactory):
    def create_calibration_difference_function(
        self, info=dict
    ) -> CalibrationDifferenceFunction:
        return ComplexInvariantDistanceFunction(info=info)


class ComplexInvariantDistanceFunction(CalibrationDifferenceFunction):
    def calculate(self, a: Any, b: Any, debug: bool, **kwargs):
        """Calculate the complex invariant distance between two time series.

        Args:
            a (Any): Vector, array
            b (Any): Vector, array
            debug (bool): [description]

        Returns:
            float: Complex invariant distance
        """
        if a.isnull().values.any():
            # print("There are missing values for CID")
            a = a.dropna()
            b = pd.Series(
                [value for value, i in zip(b, range(len(b))) if i in a.index],
                index=a.index,
            )

        combined_data = np.array([a, b])
        # get distance per datapoint
        dist = (calculate_cid(combined_data)[0][1])#/len(a)
        # print("Calculated Complex Invariant Distance of {0}".format(dist))

        return dist
