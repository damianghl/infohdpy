from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Tuple

class BaseEntropyEstimator(ABC):
    @abstractmethod
    def estimate_entropy(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> Union[float, Tuple[float, float]]:
        """
        Estimate the entropy of the given samples.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            Union[float, Tuple[float, float]]: Estimated entropy, or a tuple containing the estimated entropy and the error.
        """
        pass

class BaseMutualInformationEstimator(ABC):
    @abstractmethod
    def estimate_mutual_information(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> Union[float, Tuple[float, float]]:
        """
        Estimate the mutual information of the given samples.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            Union[float, Tuple[float, float]]: Estimated mutual information, or a tuple containing the estimated mutual information and the error.
        """
        pass

class BaseEstimator(BaseEntropyEstimator, BaseMutualInformationEstimator):
    pass