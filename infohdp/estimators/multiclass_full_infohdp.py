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
        
        # TODO: aca estamos
        
        logLbz = MulticlassInfoHDPEstimator.logprob_beta_multiclass(b1, qye, nxy)
        #ebd, ebu = self.intEb(bz, kx, n10, 3, noprior) # TODO: implement intEbT method with its corresponding D2expblogLT
        ebd, ebu = np.log(b1)-3., np.log(b1)+3. # guessing here
        listEb = np.linspace(ebd, ebu, 25)
        #listLogL = np.exp(MulticlassInfoHDPEstimator.logprob_beta_multiclass(np.exp(listEb), qye, nxy) - logLbz)
        listLogL = np.exp(np.array([MulticlassInfoHDPEstimator.logprob_beta_multiclass(np.exp(eb), qye, nxy) for eb in listEb]) - logLbz)
        listLogL /= np.sum(listLogL)
        
        sint = np.sum([MulticlassInfoHDPEstimator.conditional_entropy_hyx_multiclass(np.exp(eb), nn, qye, nxy) * ll for eb, ll in zip(listEb, listLogL)])
        #s2int = np.sum([self.SYconX2T(np.exp(eb), nn, qye, nxy) * ll for eb, ll in zip(listEb, listLogL)]) # TODO: implement SYconX2T method
        #dsint = np.sqrt(s2int - sint**2)
        dsint = 0.0
        
        ihdp = sy - sint
        return ihdp, dsint

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
        kx, Ny = nxy.shape
        dss = 0
        #return (1 / (nn**2)) * sum((n1 + n0)**2 * self.varSYx(bb, n0, n1) for n1, n0 in n10)

        for i in range(kx):
            ni_sum = np.sum(nxy[i])
            dss += (ni_sum / nn)**2 * self.varSYxT(bb, qy, nxy[i])
        return dss


    @staticmethod
    def varSYxT(b, qy, nxyj):
        """
        Calculates the variance of S(Y|x) for fixed beta and a specific state x with counts n_xy.

        Args:
            b (float): Beta value.
            nxyj (np.ndarray): Counts for state xj for different y=0,1,..., Ny-1.

        Returns:
            float: Variance of S(Y|x).
        """
        n = np.sum(nxyj)

        hYx =  special.digamma(n + b + 1) - np.sum((b * qy + nxyj) * special.digamma(1 + b * qy + nxyj)) / (n + b)

        termJ = np.sum((b * qy + nxyj)(b * qy + nxyj + 1) * self.auxJy(b, n, qy, nxyj)) / ((n + b)*(n + b + 1))
        termI = 0.
        # termI = Sum_{y1!=y2} (b * qy1 + nxy1)(b * qy2 + nxy2 ) * self.auxIyy(b, n, qy1, nxy1, qy2, nxy2)
        # termI /= ((n + b)*(n + b + 1))
        
        return termI + termJ - hYx**2

    @staticmethod
    def auxIyy(b, nx, qy1, nxy1, qy2, nxy2):
        ff = 0.
        ff += (special.digamma(nxy1+b*qy1+1)-special.digamma(nx+b+2))(special.digamma(nxy2+b*qy2+1)-special.digamma(nx+b+2))
        ff += -special.polygamma(1, nx + b + 2)
        return ff

    @staticmethod
    def auxJy(b, nx, qy, nxyj):
        """
        Calculates auxiliary function J_y as an array in y

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            qy (np.ndarray): Marginal distribution for Y.
            nxyj (np.ndarray): Counts for state xj for different y=0,1,..., Ny-1.

        Returns:
            _type_: _description_
        """
        ff += (special.digamma(nxyj + b * qy + 2) - special.digamma(nx+b+2))**2
        ff += special.polygamma(1, nxyj + b * qy + 2) - special.polygamma(1, nx + b + 2)
        return ff
