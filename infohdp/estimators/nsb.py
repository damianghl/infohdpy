# infohdp/estimators/nsb.py

import numpy as np
from typing import Union, List, Tuple
from .base import BaseEstimator
import ndd

class NSBEstimator(BaseEstimator):
    def estimate_entropy(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> Tuple[float, float]:
        """
        Estimate the entropy of the given samples using the NSB (ndd package, w/ infinite states) method.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            Tuple[float, float]: Estimated entropy and its standard deviation.
        """
        unique, counts = np.unique(samples, return_counts=True, axis=0)
        H, std_H = ndd.entropy(counts, return_std=True)
        #H, std_H = ndd.entropy(counts, k=Ns, return_std=True) #given k: number of states # TODO: maybe include option to provide K
        return H, std_H

    def estimate_mutual_information(self, samples: Union[np.ndarray, List[Tuple[int, int]]]) -> Tuple[float, float]:
        """
        Estimate the mutual information of the given samples using the NSB (ndd package, w/ infinite states) method.

        Args:
            samples (Union[np.ndarray, List[Tuple[int, int]]]): Input samples.

        Returns:
            Tuple[float, float]: Estimated mutual information and its standard deviation
        """
        nn = len(samples)
        if isinstance(samples[0], tuple):
            samxz = [s[0] for s in samples]
            samyz = [s[1] for s in samples]
        else:
            samxz = np.abs(samples)
            samyz = np.sign(samples)
        
        Hx, dHx = self.estimate_entropy(samxz)
        Hy, dHy = self.estimate_entropy(samyz)
        Hxy, dHxy = self.estimate_entropy(samples)
        
        Ixy = Hx + Hy - Hxy
        dIxy = np.sqrt(dHx**2 + dHy**2 + dHxy**2)
        
        return Ixy, dIxy
