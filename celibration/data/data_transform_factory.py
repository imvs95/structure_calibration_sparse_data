""" Defines the building blocks for creating Data Transformation object(s)
"""

import pandas as pd
from abc import ABC, abstractmethod
from typing import List
from .data_transform import DataTransform


class DataTransformFactory(ABC):
    @abstractmethod
    def create_transform(self, info=dict) -> DataTransform:
        """Abstract method that defines the creation of a DataTransform instance

        Args:
            info (dict): additional data information that is passed thtrough a key-value data structure

        Returns:
            DataTransform: DataTransform object
        """
        return NotImplementedError
