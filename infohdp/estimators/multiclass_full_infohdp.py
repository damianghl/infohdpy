import numpy as np
from scipy import stats, special, optimize
from typing import List, Tuple, Union
from .base import BaseMutualInformationEstimator
from ..utils import count_nxy_multiclass
from ..core import entropy_true
from .multiclass_infohdp import MulticlassInfoHDPEstimator

class MulticlassFullInfoHDPEstimator(BaseMutualInformationEstimator):
    def estimate_mutual_information(self, sam: Union[np.ndarray, List[Tuple[int, int]]], ML: int = 0) -> Tuple[float, float]:
        """
        Calculates the InfoHDP estimator and its error for mutual information by integrating over the peak of the posterior (only in beta).

        This method provides an estimate of mutual information based on the InfoHDP approach.

        Args:
            sam (List[Tuple[int, int]]): List of samples, where each sample is a tuple (x, y).
            ML (int, optional): If 1, use Maximum Likelihood estimation for qy; if 0, use posterior mean. Defaults to 0.

        Returns:
            Tuple[float, float]: Estimated mutual information and its standard deviation.
        """

        nn = len(sam)
        distinct_second_elements = {s[1] for s in sam}
        ny = len(distinct_second_elements)
        nxy = count_nxy_multiclass(sam)

        if ML == 1:
            qye = np.sum(nxy, axis=0) / np.sum(nxy)
        else:
            qye = (np.sum(nxy, axis=0) + 1/ny) / (np.sum(nxy) + 1)
        
        b1 = MulticlassInfoHDPEstimator.beta_solve_multiclass(qye, nxy)
        sy = entropy_true(qye)
               
        logLbz = MulticlassInfoHDPEstimator.logprob_beta_multiclass(b1, qye, nxy)
        ebd, ebu = self.intEbT(b1, qye, nxy, 3)
        listEb = np.linspace(ebd, ebu, 25)
        listLogL = np.exp(np.array([MulticlassInfoHDPEstimator.logprob_beta_multiclass(np.exp(eb), qye, nxy) for eb in listEb]) - logLbz)
        listLogL /= np.sum(listLogL) #probabilities vector seems normalized, cool
        
        sint = np.sum([MulticlassInfoHDPEstimator.conditional_entropy_hyx_multiclass(np.exp(eb), nn, qye, nxy) * ll for eb, ll in zip(listEb, listLogL)])
        s2int = np.sum([self.SYconX2T(np.exp(eb), nn, qye, nxy) * ll for eb, ll in zip(listEb, listLogL)])
        dsint = np.sqrt(s2int - sint**2)
        
        ihdp = sy - sint
        return ihdp, dsint

    @staticmethod
    def D2expblogLT(eb: float, qy: np.ndarray, nxy: np.ndarray):
        """
        Calculates the second derivative of log-likelihood of beta with respect to log(beta).

        Args:
            eb (float): Exponential of beta value.
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Second derivative of log-likelihood.
        """
        nx = np.sum(nxy, axis=1)
        b = np.exp(eb)

        ll2 = 0.
        ll2 = np.sum( b*(special.digamma(b)-special.digamma(b + nx)) + b**2 *(special.polygamma(1, b)-special.polygamma(1, b + nx)))

        for nxyi in nxy:
            nxi = np.sum(nxyi)
            ll2 += b* np.sum(qy* (special.digamma(nxyi + b * qy) - special.digamma(b * qy)))
            ll2 += b**2 * np.sum(qy**2 * (special.polygamma(1, nxyi + b * qy) - special.polygamma(1, b * qy)))
        
        return ll2

    def intEbT(self, bx: float , qy: np.ndarray, nxy: np.ndarray, nsig=3):
        """
        Calculates the interval for integration in log(beta).

        Args:
            bx (float): Beta value.
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.
            nsig (float, optional): Number of standard deviations. Defaults to 3.

        Returns:
            Tuple[float, float]: Lower and upper bounds of the integration interval.
        """
        sigeb = np.sqrt(-self.D2expblogLT(np.log(bx), qy, nxy))
        ebd = np.log(bx) - nsig * sigeb
        ebu = np.log(bx) + nsig * sigeb
        return ebd, ebu


    def SYconX2T(self, bb: float, nn: int, qy: np.ndarray, nxy: np.ndarray):
        """
        Calculates the second moment of S(Y|X) for fixed beta.

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Second moment of S(Y|X).
        """
        return self.varSYconXT(bb, nn, qy, nxy) + (MulticlassInfoHDPEstimator.conditional_entropy_hyx_multiclass(bb, nn, qy, nxy))**2 

    def varSYconXT(self, bb: float, nn: int, qy: np.ndarray, nxy: np.ndarray):
        """
        Calculates the variance of S(Y|X) for fixed beta.

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Variance of S(Y|X).
        """
        dss = 0

        for nxyi in nxy:
            ni_sum = np.sum(nxyi)
            dss += (ni_sum / nn)**2 * self.varSYxT(bb, qy, nxyi)
        return dss

    def varSYxT(self, b: float, qy: np.ndarray, nxyj: np.ndarray):
        """
        Calculates the variance of S(Y|x) for fixed beta and a specific state x with counts n_xy.

        Args:
            b (float): Beta value.
                        qy (np.ndarray): Marginal distribution for Y.
            nxyj (np.ndarray): Counts for state xj for different y=0,1,..., Ny-1.

        Returns:
            float: Variance of S(Y|x).
        """
        n = np.sum(nxyj)
        r = n + b
        ry =  b * qy + nxyj

        hYx =  special.digamma(r + 1) - np.sum(ry * special.digamma(1 + ry)) / r

        termDiag = self.auxDiag(r, ry)
        termI = self.auxIyy(r, ry)
        
        return termI + termDiag - hYx**2

    @staticmethod
    def auxIyy(r: float, ry: np.ndarray):
        """
        Calculates auxiliary function for the cross term of H^2(Y|x)

        Args:
            r (float): Some parameter, usually r = beta+ nx.
            ry (np.ndarray): Input vector, usually ry = beta * qy+ (nx)y.

        Returns:
            float: resutl of the cross term of H^2(Y|x)
        """
        ry2 = ry *(special.digamma(ry + 1) - special.digamma(r+2))
        Iyy1 = np.outer(ry2 , ry2)
        Iyy2 = np.outer(ry, ry)
        Iyy2 *= special.polygamma(1, r + 2)
        Iyy = np.sum(Iyy1 - Iyy2)
        Iyy /= r*(r + 1)
        return Iyy

    @staticmethod
    def auxDiag(r: float, ry: np.ndarray):
        """
        Calculates auxiliary function for the diagonal term of H^2(Y|x)

        Args:
            r (float): Some parameter, usually r = beta+ nx.
            ry (np.ndarray): Input vector, usually ry = beta * qy+ (nx)y.

        Returns:
            float: resutl of the diagonal term of H^2(Y|x)
        """
        Jy = (special.digamma(ry + 2) - special.digamma(r+2))**2
        Jy += special.polygamma(1, ry + 2) - special.polygamma(1, r + 2)
        Jy *= ry*(ry + 1)
        diag = np.sum(Jy)
        diag /= r*(r + 1)
        return diag
