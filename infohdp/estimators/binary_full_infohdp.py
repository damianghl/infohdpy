import numpy as np
from scipy import stats, special, optimize
from typing import List, Tuple, Union
from .base import BaseMutualInformationEstimator
from ..utils import freq_of_frequencies, count_nxy_binary
from ..core import entropy_true
from .binary_infohdp import BinaryInfoHDPEstimator

class BinaryFullInfoHDPEstimator(BaseMutualInformationEstimator):
    def estimate_mutual_information(self, sam: Union[np.ndarray, List[Tuple[int, int]]], onlyb: int = 0, noprior: int = 0, ML: int = 1) -> Tuple[float, float]:
        """
        Calculates the InfoHDP estimator and its error for mutual information by integrating over the peak of the posterior (only in beta).

        This method provides an estimate of mutual information based on the InfoHDP approach.

        Args:
            sam (Union[np.ndarray, List[Tuple[int, int]]]): Sample data.
            onlyb (int, optional): If 1, uses only beta (no alpha, i.e., no pseudocounts). Defaults to 0.
            noprior (int, optional): If 1, no prior is used for beta. Defaults to 0.

        Returns:
            Tuple[float, float]: Estimated mutual information and its standard deviation.
        """
        # nn = len(sam)
        # a1 = 0
        
        # if onlyb != 1:
        #     kk = len(np.unique(sam))
        #     a1 = BinaryInfoHDPEstimator.alpha_solve(nn, kk)  # Note: You need to implement asol method or import it
        
        # samx = np.abs(sam)
        # kx = len(np.unique(samx))
        # n10 = count_nxy_binary(sam)
        
        # if ML == 1:
        #     qye = np.sum(n10, axis=0) / np.sum(n10)
        # else:
        #     qye = (np.sum(n10, axis=0) + 1/2) / (np.sum(n10) + 1)
        
        # b1 = BinaryInfoHDPEstimator.beta_solve(kx, n10, noprior) 
        # sy = entropy_true(qye)
        # sycx = BinaryInfoHDPEstimator.conditional_entropy_hyx(a1, b1, nn, n10)
        
        # ihdp = sy - sycx

        nn = len(sam)
        az = 0
        
        if onlyb != 1:
            kk = len(np.unique(sam))
            az = BinaryInfoHDPEstimator.alpha_solve(nn, kk)
        
        samx = np.abs(sam)
        kx = len(np.unique(samx))
        n10 = count_nxy_binary(sam)

        if ML == 1:
            qye = np.sum(n10, axis=0) / np.sum(n10)
        else:
            qye = (np.sum(n10, axis=0) + 1/2) / (np.sum(n10) + 1)
        
        bz = BinaryInfoHDPEstimator.beta_solve(kx, n10, noprior) 
        sy = entropy_true(qye)
        
        logLbz = BinaryInfoHDPEstimator.logprob_beta(bz, kx, n10, noprior)
        ebd, ebu = self.intEb(bz, kx, n10, 3, noprior)
        listEb = np.linspace(ebd, ebu, 25)
        listLogL = np.exp(BinaryInfoHDPEstimator.logprob_beta(np.exp(listEb), kx, n10, noprior) - logLbz)
        listLogL /= np.sum(listLogL)
        
        sint = np.sum([BinaryInfoHDPEstimator.conditional_entropy_hyx(az, np.exp(eb), nn, n10) * ll for eb, ll in zip(listEb, listLogL)])
        s2int = np.sum([self.SYconX2(np.exp(eb), nn, n10) * ll for eb, ll in zip(listEb, listLogL)]) # TODO: aca me quede
        dsint = np.sqrt(s2int - sint**2)
        
        ihdp = sy - sint
        return ihdp, dsint, sint
    

    @staticmethod
    def D2expblogL(eb, kx, n10, noprior=0):
        """
        Calculates the second derivative of log-likelihood of beta with respect to log(beta).

        Args:
            eb (float): Exponential of beta value.
            kx (int): Number of unique X samples.
            n10 (List[List[int]]): n10 statistics.
            noprior (int, optional): If 1, no prior is used. Defaults to 0.

        Returns:
            float: Second derivative of log-likelihood.
        """
        exp_eb = np.exp(eb)
        result = kx * (2 * exp_eb * special.digamma(2 * exp_eb) - 
                    2 * (exp_eb * special.digamma(exp_eb) + exp_eb**2 * special.polygamma(1, exp_eb)) + 
                    4 * exp_eb**2 * special.polygamma(1, 2 * exp_eb))
        
        for ni1, ni0 in n10:
            result += (-2 * exp_eb * special.digamma(2 * exp_eb + ni1 + ni0) +
                    exp_eb * special.digamma(exp_eb + ni0) +
                    exp_eb * special.digamma(exp_eb + ni1) -
                    4 * exp_eb**2 * special.polygamma(1, 2 * exp_eb + ni1 + ni0) +
                    exp_eb**2 * special.polygamma(1, exp_eb + ni0) +
                    exp_eb**2 * special.polygamma(1, exp_eb + ni1))
        
        if noprior != 1:
            result += (-(exp_eb * special.polygamma(2, 1 + exp_eb) - 4 * exp_eb * special.polygamma(2, 1 + 2 * exp_eb))**2 /
                    (-special.polygamma(1, 1 + exp_eb) + 2 * special.polygamma(1, 1 + 2 * exp_eb))**2 +
                    (-exp_eb * special.polygamma(2, 1 + exp_eb) + 4 * exp_eb * special.polygamma(2, 1 + 2 * exp_eb) -
                        exp_eb**2 * special.polygamma(3, 1 + exp_eb) + 8 * exp_eb**2 * special.polygamma(3, 1 + 2 * exp_eb)) /
                    (-special.polygamma(1, 1 + exp_eb) + 2 * special.polygamma(1, 1 + 2 * exp_eb)))
        
        return result

    def intEb(self, bx, kx, n10, nsig=3, noprior=0):
        """
        Calculates the interval for integration in log(beta).

        Args:
            bx (float): Beta value.
            kx (int): Number of unique X samples.
            n10 (List[List[int]]): n10 statistics.
            nsig (float, optional): Number of standard deviations. Defaults to 3.
            noprior (int, optional): If 1, no prior is used. Defaults to 0.

        Returns:
            Tuple[float, float]: Lower and upper bounds of the integration interval.
        """
        sigeb = np.sqrt(-self.D2expblogL(np.log(bx), kx, n10, noprior))
        ebd = np.log(bx) - nsig * sigeb
        ebu = np.log(bx) + nsig * sigeb
        return ebd, ebu

    def SYconX2(self, bb, nn, n10):
        """
        Calculates the second moment of S(Y|X) for fixed beta.

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            n10 (List[List[int]]): n10 statistics.

        Returns:
            float: Second moment of S(Y|X).
        """
        return self.varSYconX(bb, nn, n10) + (BinaryInfoHDPEstimator.conditional_entropy_hyx(0., bb, nn, n10))**2

    def varSYconX(self, bb, nn, n10):
        """
        Calculates the variance of S(Y|X) for fixed beta.

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            n10 (List[List[int]]): n10 statistics.

        Returns:
            float: Variance of S(Y|X).
        """
        return (1 / (nn**2)) * sum((n1 + n0)**2 * self.varSYx(bb, n0, n1) for n1, n0 in n10)

    @staticmethod
    def varSYx(b, n0, n1):
        """
        Calculates the variance of S(Y|x) for fixed beta and a specific state x with counts n_x = n0 + n1.

        Args:
            b (float): Beta value.
            n0 (int): Count for state 0.
            n1 (int): Count for state 1.

        Returns:
            float: Variance of S(Y|x).
        """
        n = n0 + n1
        term1 = (2 * (b + n1) * (b + n0)) / ((2*b + n) * (2*b + n + 1))
        term1 *= ((special.digamma(b + n1 + 1) - special.digamma(2*b + n + 2)) *
                  (special.digamma(b + n0 + 1) - special.digamma(2*b + n + 2)) -
                  special.polygamma(1, 2*b + n + 2))
        
        term2 = ((b + n1) * (b + n1 + 1)) / ((2*b + n) * (2*b + n + 1))
        term2 *= ((special.digamma(b + n1 + 2) - special.digamma(2*b + n + 2))**2 +
                  special.polygamma(1, b + n1 + 2) - special.polygamma(1, 2*b + n + 2))
        
        term3 = ((b + n0) * (b + n0 + 1)) / ((2*b + n) * (2*b + n + 1))
        term3 *= ((special.digamma(b + n0 + 2) - special.digamma(2*b + n + 2))**2 +
                  special.polygamma(1, b + n0 + 2) - special.polygamma(1, 2*b + n + 2))
        
        term4 = (special.digamma(2*b + n + 1) -
                 (b + n1) / (2*b + n) * special.digamma(b + n1 + 1) -
                 (b + n0) / (2*b + n) * special.digamma(b + n0 + 1))**2
        
        return term1 + term2 + term3 - term4

    @staticmethod
    def SYx2(b, n0, n1):
        """
        Calculates the second moment of S(Y|x) for fixed beta and a specific state x with counts n_x = n0 + n1.

        Args:
            b (float): Beta value.
            n0 (int): Count for state 0.
            n1 (int): Count for state 1.

        Returns:
            float: Second moment of S(Y|x).
        """
        n = n0 + n1
        term1 = (2 * (b + n1) * (b + n0)) / ((2*b + n) * (2*b + n + 1))
        term1 *= ((special.digamma(b + n1 + 1) - special.digamma(2*b + n + 2)) *
                  (special.digamma(b + n0 + 1) - special.digamma(2*b + n + 2)) -
                  special.polygamma(1, 2*b + n + 2))
        
        term2 = ((b + n1) * (b + n1 + 1)) / ((2*b + n) * (2*b + n + 1))
        term2 *= ((special.digamma(b + n1 + 2) - special.digamma(2*b + n + 2))**2 +
                  special.polygamma(1, b + n1 + 2) - special.polygamma(1, 2*b + n + 2))
        
        term3 = ((b + n0) * (b + n0 + 1)) / ((2*b + n) * (2*b + n + 1))
        term3 *= ((special.digamma(b + n0 + 2) - special.digamma(2*b + n + 2))**2 +
                  special.polygamma(1, b + n0 + 2) - special.polygamma(1, 2*b + n + 2))
        
        return term1 + term2 + term3
