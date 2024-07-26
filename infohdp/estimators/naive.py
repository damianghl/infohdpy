import numpy as np
from typing import Union, List, Tuple
from .base import BaseEstimator
from ..utils import dkm2  # Import dkm2 from utils

class NaiveEstimator(BaseEstimator):
    @staticmethod
    def snaive(nn: int, dkm2: List[Tuple[int, int]]) -> float:
        """
        Compute naive entropy estimate.
        
        Args:
            nn (int): Total number of samples.
            dkm2 (List[Tuple[int, int]]): Frequency of frequencies.
        
        Returns:
            float: Naive entropy estimate.
        """
        return -sum(count * (freq / nn) * np.log(freq / nn) for freq, count in dkm2)

    def estimate_entropy(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> float:
        """
        Estimate the entropy of the given samples using the naive method.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            float: Estimated entropy.
        """
        nn = len(samples)
        dkm2_result = dkm2(samples)
        return self.snaive(nn, dkm2_result)

    def estimate_mutual_information(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> float:
        """
        Estimate the mutual information of the given samples using the naive method.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            float: Estimated mutual information.
        """
        nn = len(samples)
        if isinstance(samples[0], tuple):
            samxz = [s[0] for s in samples]
            samyz = [s[1] for s in samples]
        else:
            samxz = np.abs(samples)
            samyz = np.sign(samples)
        
        dkmz = dkm2(samples)
        dkmzX = dkm2(samxz)
        dkmzY = dkm2(samyz)
        
        return (self.snaive(nn, dkmzX) + 
                self.snaive(nn, dkmzY) - 
                self.snaive(nn, dkmz))
