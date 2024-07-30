import numpy as np
from scipy import stats, special, optimize
from typing import List, Tuple, Union
from .base import BaseMutualInformationEstimator
from ..utils import nxysam

class MulticlassInfoHDPEstimator(BaseMutualInformationEstimator):
    @staticmethod
    def bsolT(qy: np.ndarray, nxy: np.ndarray) -> float:
        """
        Gives the beta that maximizes the marginal log-likelihood, given an estimated qy and counts.

        Args:
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Optimal beta value.
        """
        def objective(ebb):
            return -MulticlassInfoHDPEstimator.logLbT(np.exp(ebb), qy, nxy)
        
        result = optimize.minimize_scalar(objective, bounds=(-10, 10), method='bounded')
        return np.exp(result.x)

    @staticmethod
    def logLbT(b: float, qy: np.ndarray, nxy: np.ndarray) -> float:
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
    def SYconXT(bb: float, nn: int, qy: np.ndarray, nxy: np.ndarray) -> float:
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
        nxy = nxysam(sam, ny)
        
        if ML == 1:
            qye = np.sum(nxy, axis=0) / np.sum(nxy)
        else:
            qye = (np.sum(nxy, axis=0) + 1/ny) / (np.sum(nxy) + 1)
        
        b1 = self.bsolT(qye, nxy)
        sy = self.strue(qye)
        sycx = self.SYconXT(b1, nn, qye, nxy)
        ihdp = sy - sycx
        return ihdp

    @staticmethod # TODO: call instead from core
    def strue(p: np.ndarray) -> float:
        """
        Implementation of Strue (true entropy)

        Args:
            p (np.ndarray): Probability distribution.

        Returns:
            float: True entropy.
        """
        # Ensure p is a numpy array
        p = np.asarray(p)
        
        # Compute the entropy safely
        entropy = -np.sum(p * np.log(p, where=(p > 0), out=np.zeros_like(p)))
        return entropy