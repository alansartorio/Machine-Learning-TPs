from abc import ABC, abstractmethod

import numpy as np


class Standardization(ABC):
    @abstractmethod
    def standardize(self, values: np.ndarray) -> np.ndarray:
        pass


class UnitLengthScaling(Standardization):
    def standardize(self, values: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(values)
        return np.divide(values, norm)


class MinMaxFeatureScaling(Standardization):
    def __init__(self, range=(0.0, 1.0)):
        self.range = range

    def standardize(self, values: np.ndarray) -> np.ndarray:
        min = np.ones(values.shape) * np.min(values)
        max = np.ones(values.shape) * np.max(values)
        return ((values - min) / (max - min)) * (
            self.range[1] - self.range[0]
        ) + self.range[0]


class ZScore(Standardization):
    def standardize(self, values: np.ndarray) -> np.ndarray:
        mean = np.mean(values)
        std_dev = np.std(values)

        if std_dev == 0:
            raise Exception("Cannot standardize data with zero standard deviation")

        standardized_values = (values - mean) / std_dev

        return standardized_values
