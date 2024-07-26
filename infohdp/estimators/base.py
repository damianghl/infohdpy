# infohdp/estimators/base.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List, Tuple

class BaseEstimator(ABC):
    @abstractmethod
    def estimate_entropy(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> float:
        """
        Estimate the entropy of the given samples.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            float: Estimated entropy.
        """
        pass

    @abstractmethod
    def estimate_mutual_information(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> float:
        """
        Estimate the mutual information of the given samples.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            float: Estimated mutual information.
        """
        pass

    @staticmethod
    def dkm2(sam: np.ndarray) -> List[Tuple[int, int]]:
        """
        Compute frequency of frequencies.
        
        Args:
            sam (np.ndarray): Sample data.
        
        Returns:
            List[Tuple[int, int]]: Frequency of frequencies.
        """
        unique, counts = np.unique(sam, return_counts=True)
        unique_counts, count_counts = np.unique(counts, return_counts=True)
        return sorted(zip(unique_counts, count_counts))
