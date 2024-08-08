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
        #s2int = np.sum([self.SYconX2(az, np.exp(eb), nn, n10) * ll for eb, ll in zip(listEb, listLogL)]) # TODO: implement SYconX2T method
        #dsint = np.sqrt(s2int - sint**2)
        dsint = 0.0
        
        ihdp = sy - sint
        return ihdp, dsint
