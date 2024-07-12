""" Defines the building blocks for creating Calibration Difference Dunction object(s)
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from .calibration_difference_function import CalibrationDifferenceFunction


class CalibrationDifferenceFunctionFactory(ABC):
    def __init__(self, info):
        self._info = info
    
    def get_info(self):
        return self._info

    @abstractmethod
    def create_calibration_difference_function(self) -> CalibrationDifferenceFunction:
        """Abstract method that defines the creation of a calibration difference

        Returns:
            CalibrationDifferenceFunction: CalibrationDifferenceFunction object
        """
        return NotImplementedError
