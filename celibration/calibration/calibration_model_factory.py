""" Defines the building blocks for creating calibration model object(s)
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from .calibration_model import CalibrationModel


class CalibrationModelFactory(ABC):
    @abstractmethod
    def create_calibration_model(self, info=dict) -> CalibrationModel:
        """Abstract method that defines the creation of a calibration model

        Args:
            info (dict): additional model information that is passed thtrough a key-value data structure

        Returns:
            CalibrationModel: CalibrationModel object
        """
        return NotImplementedError
