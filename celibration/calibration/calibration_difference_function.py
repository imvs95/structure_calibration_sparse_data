""" Abstract definition of the building blocks of a CalibrationDifferenceFunction object
"""

from typing import Any
from abc import ABC, abstractmethod


class CalibrationDifferenceFunction(ABC):
    def __init__(self, info):
        self._info = info

    def get_info(self):
        return self._info

    @abstractmethod
    def calculate(self, a: Any, b: Any, debug: bool, **kwargs):
        """Abstract method that defines a calculation given a difference function and two types of input

        Args:
            a (Any): Object that holds a sole integer, float or a list containing either one
            b (Any): Object that holds a sole integer, float or a list containing either one
            debug (bool): Boolean indicating if logging verbosity should be applied
        Returns:
            [type]: [description]
        """
        return NotImplementedError
