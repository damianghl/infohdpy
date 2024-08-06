import numpy as np
from scipy import stats, special, optimize
from typing import List, Tuple, Union
from .base import BaseMutualInformationEstimator
from ..utils import count_nxy_multiclass
from ..core import entropy_true

class MulticlassInfoHDPEstimator(BaseMutualInformationEstimator):
    @staticmethod
    def beta_solve_multiclass(qy: np.ndarray, nxy: np.ndarray) -> float:
        """
        Gives the beta that maximizes the marginal log-likelihood, given an estimated qy and counts.

        Args:
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Optimal beta value.
        """
        def objective(ebb):
            return -MulticlassInfoHDPEstimator.logprob_beta_multiclass(np.exp(ebb), qy, nxy)
        
        result = optimize.minimize_scalar(objective, bounds=(-10, 10), method='bounded')
        return np.exp(result.x)

    @staticmethod
    def logprob_beta_multiclass(b: float, qy: np.ndarray, nxy: np.ndarray) -> float:
        """
        Gives the marginal log-likelihood for beta, given a marginal (estimated) qy.

        Args:
            b (float): Beta value.
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Log-likelihood value.
        """
        kx, Ny = nxy.shape
        ll = kx * (special.gammaln(b) - np.sum(special.gammaln(b * qy)))
        ll += np.sum(np.sum(special.gammaln(b * qy + nxy), axis=1) - special.gammaln(b + np.sum(nxy, axis=1)))
        return ll

    @staticmethod
    def conditional_entropy_hyx_multiclass(bb: float, nn: int, qy: np.ndarray, nxy: np.ndarray) -> float:
        """
        Gives the posterior for the conditional entropy S(Y|X).

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Posterior conditional entropy S(Y|X).
        """
        kx, Ny = nxy.shape
        ss = 0
        for i in range(kx):
            ni_sum = np.sum(nxy[i])
            ss += (ni_sum / nn) * (special.digamma(ni_sum + bb + 1) - 
                                np.sum((bb * qy + nxy[i]) * special.digamma(1 + bb * qy + nxy[i])) / (ni_sum + bb))
        return ss

    def estimate_mutual_information(self, sam: List[Tuple[int, int]], ML: int = 0) -> float:
        """
        Gives the MAP estimate of mutual information using InfoHDP.

        Args:
            sam (List[Tuple[int, int]]): List of samples, where each sample is a tuple (x, y).
            ML (int, optional): If 1, use Maximum Likelihood estimation for qy; if 0, use posterior mean. Defaults to 0.

        Returns:
            float: Estimated mutual information.
        """
        nn = len(sam)
        distinct_second_elements = {s[1] for s in sam}
        # Calculate the number of distinct elements
        ny = len(distinct_second_elements)
        nxy = count_nxy_multiclass(sam)
        
        if ML == 1:
            qye = np.sum(nxy, axis=0) / np.sum(nxy)
        else:
            qye = (np.sum(nxy, axis=0) + 1/ny) / (np.sum(nxy) + 1)
        
        b1 = self.beta_solve_multiclass(qye, nxy)
        sy = entropy_true(qye)
        sycx = self.conditional_entropy_hyx_multiclass(b1, nn, qye, nxy)
        ihdp = sy - sycx
        return ihdp
