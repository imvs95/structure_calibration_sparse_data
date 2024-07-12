import pandas as pd
import numpy as np

from typing import Any
from celibration import (
    CalibrationDifferenceFunction,
    CalibrationDifferenceFunctionFactory,
)

from sklearn.metrics.pairwise import cosine_similarity


class PluginFactory(CalibrationDifferenceFunctionFactory):
    def create_calibration_difference_function(
        self, info=dict
    ) -> CalibrationDifferenceFunction:
        return CosineSimilarityFunction(info=info)
        return CosineSimilarityFunction(info=info)


class CosineSimilarityFunction(CalibrationDifferenceFunction):
    def calculate(self, a: Any, b: Any, debug: bool, **kwargs):
        """Compute cosine similarity between samples in a and b. We maximize the Cosine Similarity, this is
        similar to minimizing the Cosine Distance. Thus, we use the Cosine Distance here, which is 1-Cosine
        Similarity.

        Args:
            a (Any): Vector, array
            b (Any): Vector, array
            debug (bool): [description]

        Returns:
            float: Cosine similarity as distance
        """
        if a.isnull().values.any():
            # print("There are missing values for Cosine Similarity")
            a = a.dropna()
            b = pd.Series(
                [value for value, i in zip(b, range(len(b))) if i in a.index],
                index=a.index,
            )

        dist_similarity = cosine_similarity([a], [b])[0][0]
        # This is the Cosine distance which we want to minimize,
        # Else maximize the Cosine similarity
        # Get distance per data point
        dist = (1 - dist_similarity) #/ len(a)
        # print("Calculated (1-)Cosine Similarity of {0}".format(dist))

        return dist
