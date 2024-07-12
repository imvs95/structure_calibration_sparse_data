import pandas as pd
import math
from random import randint
from typing import Any
from celibration import (
    CalibrationDifferenceFunction,
    CalibrationDifferenceFunctionFactory,
)


class PluginFactory(CalibrationDifferenceFunctionFactory):
    def create_calibration_difference_function(
        self, info: dict
    ) -> CalibrationDifferenceFunction:
        return ExampleDifferenceFunction(info=info)


class ExampleDifferenceFunction(CalibrationDifferenceFunction):
    def calculate(self, a: Any, b: Any, debug: bool, **kwargs):
        """Calculate the distance between a and b.

        Args:
            a (Any): Vector, array
            b (Any): Vector, array
            debug (bool): [description]

        Returns:
            float: Distance
        """
        if debug:
            print("ExampleCalibrationDifferenceFunction")
            print(f"\tParameters: {kwargs}")
        try:
            dist = a - b
        except:
            dist = -10 + randint(0, 20)
        return dist
