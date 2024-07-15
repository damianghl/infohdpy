import numpy as np
from scipy import stats, special
from scipy.optimize import minimize
from typing import List, Tuple

class InfoHDP:
    @staticmethod
    def strue(p):
        # Implementation of Strue
        return -np.sum(np.where(p > 0, p * np.log(p), 0))

    @staticmethod
    def sxtrue(p):
        # Implementation of Sxtrue
        p_reshaped = p.reshape(-1, 2)
        return InfoHDP.strue(np.sum(p_reshaped, axis=1))

    @staticmethod
    def sytrue(p):
        # Implementation of Sytrue
        p_reshaped = p.reshape(-1, 2)
        return InfoHDP.strue(np.sum(p_reshaped, axis=0))

    @staticmethod
    def itrue(p):
        # Implementation of Itrue
        return -InfoHDP.strue(p) + InfoHDP.sytrue(p) + InfoHDP.sxtrue(p)
    
    @staticmethod
    def gen_prior_pij(alpha: float, beta: float, ndist: int = 1, Ns: int = 10000) -> np.ndarray:
        """
        Generate probability distributions.
        
        Args:
            alpha (float): Concentration parameter for Dirichlet process.
            beta (float): Parameter for Beta distribution.
            ndist (int): Number of distributions to generate.
            Ns (int): Number of states.
        
        Returns:
            np.ndarray: Probability distributions.
        """
        alist = np.full(Ns, alpha / Ns)
        bes = stats.beta.rvs(beta, beta, size=Ns)
        pi = stats.dirichlet.rvs(alist, size=ndist)
        pi = np.column_stack((pi, 1 - np.sum(pi, axis=1)))
        pij = np.array([np.concatenate([(pi[k, i] * bes[i], pi[k, i] * (1 - bes[i])) for i in range(Ns)]) for k in range(ndist)])
        return pij

    @staticmethod
    def gen_samples_prior(qij: np.ndarray, M: int, Ns: int = 10000) -> np.ndarray:
        """
        Generate samples from prior distribution.
        
        Args:
            qij (np.ndarray): Probability distribution.
            M (int): Number of samples to generate.
            Ns (int): Number of states.
        
        Returns:
            np.ndarray: Generated samples.
        """
        name_states = np.concatenate([np.arange(1, Ns + 1), -np.arange(1, Ns + 1)])
        return np.random.choice(name_states, size=M, p=np.abs(qij.flatten()))

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

    @staticmethod
    def smaxlik(sam: np.ndarray) -> float:
        """
        Compute maximum likelihood entropy estimate.
        
        Args:
            sam (np.ndarray): Sample data.
        
        Returns:
            float: Maximum likelihood entropy estimate.
        """
        return InfoHDP.snaive(len(sam), InfoHDP.dkm2(sam))

    @staticmethod
    def inaive(sam: np.ndarray) -> float:
        """
        Compute naive mutual information estimate.
        
        Args:
            sam (np.ndarray): Sample data.
        
        Returns:
            float: Naive mutual information estimate.
        """
        nn = len(sam)
        samxz = np.abs(sam)
        samyz = np.sign(sam)
        dkmz = InfoHDP.dkm2(sam)
        dkmzX = InfoHDP.dkm2(samxz)
        dkmzY = InfoHDP.dkm2(samyz)
        return (InfoHDP.snaive(nn, dkmzX) + 
                InfoHDP.snaive(nn, dkmzY) - 
                InfoHDP.snaive(nn, dkmz))

