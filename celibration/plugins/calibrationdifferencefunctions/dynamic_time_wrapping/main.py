import pandas as pd
import numpy as np

from typing import Any
from celibration import (
    CalibrationDifferenceFunction,
    CalibrationDifferenceFunctionFactory,
)

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class PluginFactory(CalibrationDifferenceFunctionFactory):
    def create_calibration_difference_function(
        self, info=dict
    ) -> CalibrationDifferenceFunction:
        return DynamicTimeWrappingFunction(info=info)


class DynamicTimeWrappingFunction(CalibrationDifferenceFunction):
    def calculate(self, a: Any, b: Any, debug: bool, **kwargs):
        """Calculate similarity between two temporal sequences, distance in time series a and b.

        Args:
            a (Any): Vector, array
            b (Any): Vector, array
            debug (bool): [description]

        Returns:
            float: Distance between the time series
        """
        if a.isnull().values.any():
            # print("There are missing values for DTW")
            a = a.dropna()

        dist, path = fastdtw(a, b, dist=euclidean)
        # Get distance per data point
        dist = dist #/ len(a)
        # print("Calculated Dynamic Time Wrapping of {0}".format(dist))

        return dist
