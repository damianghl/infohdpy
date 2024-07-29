import numpy as np
from scipy import stats, special, optimize
from typing import List, Tuple, Union
from .base import BaseMutualInformationEstimator
from ..utils import n10sam, dkm2

class BinaryInfoHDPEstimator(BaseMutualInformationEstimator):
    @staticmethod
    def bsol(kx: int, n10: List[List[int]], noprior: float = 0.) -> float:
        """
        Solve for beta.
        
        Args:
            kx (int): Number of unique X samples.
            n10 (List[List[int]]): n10 statistics.
            noprior (float): Prior weight.
        
        Returns:
            float: Solved beta value.
        """
        def objective(log_b):
            return -BinaryInfoHDPEstimator.logLb(np.exp(log_b), kx, n10, noprior)
        
        result = optimize.minimize_scalar(objective, bounds=(-10, 10), method='bounded')
        return np.exp(result.x)

    @staticmethod
    def logLb(b: float, kx: int, n10: List[List[int]], noprior: float = 0.) -> float:
        """
        Compute log-likelihood for beta.
        
        Args:
            b (float): Beta value.
            kx (int): Number of unique X samples.
            n10 (List[List[int]]): n10 statistics.
            noprior (float): Prior weight.
        
        Returns:
            float: Log-likelihood for beta.
        """
        ll = (kx * (special.gammaln(2*b) - 2*special.gammaln(b)) + 
              sum(special.gammaln(b + n1) + special.gammaln(b + n0) - special.gammaln(2*b + n1 + n0) 
                  for n1, n0 in n10))
        if noprior != 1:
            ll += np.log(b) + np.log(2*special.polygamma(1, 2*b+1) - special.polygamma(1, b+1))
        return ll

    @staticmethod
    def SYconX(x: float, bb: float, nn: int, n10: List[List[int]]) -> float:
        """
        Compute conditional entropy S(Y|X).
        
        Args:
            x (float): Alpha value.
            bb (float): Beta value.
            nn (int): Total number of samples.
            n10 (List[List[int]]): n10 statistics.
        
        Returns:
            float: Conditional entropy S(Y|X).
        """
        return ((x / (x + nn)) * (special.polygamma(0, 2*bb+1) - special.polygamma(0, bb+1)) + 
                (1 / (x + nn)) * sum((n1 + n0) * (special.polygamma(0, n1 + n0 + 2*bb + 1) - 
                                                  (n1 + bb) / (n1 + n0 + 2*bb) * special.polygamma(0, n1 + bb + 1) -
                                                  (n0 + bb) / (n1 + n0 + 2*bb) * special.polygamma(0, n0 + bb + 1))
                                     for n1, n0 in n10))

    def estimate_mutual_information(self, sam: Union[np.ndarray, List[Tuple[int, int]]], onlyb: int = 0, noprior: int = 0) -> float:
        """
        Calculates the MAP (Maximum A Posteriori) estimate of mutual information using InfoHDP.

        This method provides an estimate of mutual information based on the InfoHDP approach.

        Args:
            sam (Union[np.ndarray, List[Tuple[int, int]]]): Sample data.
            onlyb (int, optional): If 1, uses only beta (no alpha, i.e., no pseudocounts). Defaults to 0.
            noprior (int, optional): If 1, no prior is used for beta. Defaults to 0.

        Returns:
            float: Estimated mutual information.
        """
        nn = len(sam)
        a1 = 0
        
        if onlyb != 1:
            kk = len(np.unique(sam))
            a1 = self.asol(nn, kk)  # Note: You need to implement asol method or import it
        
        samx = np.abs(sam)
        kx = len(np.unique(samx))
        n10 = n10sam(sam)
        b1 = self.bsol(kx, n10, noprior)
        
        sy = self.smaxlik(np.sign(sam))  # Note: You need to implement smaxlik method or import it
        sycx = self.SYconX(a1, b1, nn, n10)
        
        ihdp = sy - sycx
        return ihdp

    @staticmethod # FIXME: maybe unnecessary?
    def smaxlik(sam: np.ndarray) -> float:
        """
        Compute maximum likelihood entropy estimate.
        
        Args:
            sam (np.ndarray): Sample data.
        
        Returns:
            float: Maximum likelihood entropy estimate.
        """
        return BinaryInfoHDPEstimator.snaive(len(sam), dkm2(sam))

    @staticmethod # TODO: call instead from estimators/naive
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

    @staticmethod
    def asol(nn: int, k: int) -> float:
        """
        Solve for alpha (NSB).
        
        Args:
            nn (int): Total number of samples.
            k (int): Number of unique samples.
        
        Returns:
            float: Solved alpha value.
        """
        x1 = nn * (k / nn) ** (3/2) / np.sqrt(2 * (1 - k/nn))
        
        def objective(x):
            return (k - 1) / x + special.polygamma(0, 1 + x) - special.polygamma(0, nn + x)
        
        result = optimize.root_scalar(objective, x0=x1, x1=x1*1.1)
        return result.root