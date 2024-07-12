import pandas as pd
import math
from random import randint
from typing import Any
from celibration import (
    CalibrationDifferenceFunction,
    CalibrationDifferenceFunctionFactory,
)

from scipy.spatial import distance
from sklearn.metrics.pairwise import manhattan_distances


class PluginFactory(CalibrationDifferenceFunctionFactory):
    def create_calibration_difference_function(
        self, info=dict
    ) -> CalibrationDifferenceFunction:
        return ManhattanMetricsFunction(info=info)


class ManhattanMetricsFunction(CalibrationDifferenceFunction):
    def calculate(self, a: Any, b: Any, debug: bool, **kwargs):
        """Calculate the Manhattan distance between two vectors a and b.

        Args:
            a (Any): Vector, array
            b (Any): Vector, array
            debug (bool): [description]

        Returns:
            float: Manhattan distance
        """
        try:
            dist = (manhattan_distances([a], [b])[0][0])#/len(a)
            # dist = manhattan_distances(a, b)
            # print("Calculated Manhattan Distance of {0}".format(dist))
        except ValueError:
            pass

        try:
            a = a.dropna()
            b = pd.Series(
                [value for value, i in zip(b, range(len(b))) if i in a.index],
                index=a.index,
            )
            #Get distance per point in time series, else in statistics without
            dist = (manhattan_distances([a], [b])[0][0])#/len(a)
            # print(
            #     "With Missing Values: Calculated Manhattan Distance of {0}".format(dist)
            # )
        except (ValueError, AttributeError):
            pass

        return dist
