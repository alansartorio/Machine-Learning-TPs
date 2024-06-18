import math
from abc import ABC, abstractmethod

import numpy as np


class Similarity(ABC):
    @abstractmethod
    def calculate(self, input: np.ndarray, weights: np.ndarray) -> float:
        pass


class EuclideanSimilarity(Similarity):
    def calculate(self, input: np.ndarray, weights: np.ndarray) -> float:
        return np.linalg.norm(input - weights)


class ExponentialSimilarity(Similarity):
    def calculate(self, input: np.ndarray, weights: np.ndarray) -> float:
        return np.exp(math.pow(-np.linalg.norm(input - weights),2))


