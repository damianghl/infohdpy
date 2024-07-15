import numpy as np
from scipy import stats, special
from scipy.optimize import minimize
from scipy import optimize
from typing import List, Tuple, Union
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

    @staticmethod
    def logLa(x: float, n: int, k: int) -> float:
        """
        Compute the log-likelihood for alpha (NSB).
        
        Args:
            x (float): Alpha value.
            n (int): Total number of samples.
            k (int): Number of unique samples.
        
        Returns:
            float: Log-likelihood for alpha.
        """
        return (k - 1) * np.log(x) + special.gammaln(1 + x) - special.gammaln(n + x)

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

    @staticmethod
    def Spost(x: float, nn: int, dkm: List[Tuple[int, int]]) -> float:
        """
        Compute posterior entropy (NSB).
        
        Args:
            x (float): Alpha value.
            nn (int): Total number of samples.
            dkm (List[Tuple[int, int]]): Frequency of frequencies.
        
        Returns:
            float: Posterior entropy.
        """
        return (special.polygamma(0, nn + x + 1) - 
                (1 / (x + nn)) * sum(count * freq * special.polygamma(0, freq + 1) for freq, count in dkm))

    @staticmethod
    def Sint(sam: np.ndarray) -> Tuple[float, float]:
        """
        Compute NSB entropy estimate with integration.
        
        Args:
            sam (np.ndarray): Sample data.
        
        Returns:
            Tuple[float, float]: Estimated entropy and its standard deviation.
        """
        nn = len(sam)
        dkmz = InfoHDP.dkm2(sam)
        kz = len(np.unique(sam))
        az = InfoHDP.asol(nn, kz)
        
        def integrand(log_x):
            x = np.exp(log_x)
            return InfoHDP.Spost(x, nn, dkmz) * np.exp(InfoHDP.logLa(x, nn, kz))
        
        result = integrate.quad(integrand, np.log(az/10), np.log(az*10))
        sint = result[0]
        dsint = np.sqrt(result[1])  # Approximating standard deviation as sqrt of absolute error
        
        return sint, dsint

    @staticmethod
    def Insb(sam: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Compute NSB mutual information estimate.
        
        Args:
            sam (np.ndarray): Sample data.
        
        Returns:
            Tuple[float, float, float, float]: NSB MI estimate, Sx, Sy, Sxy.
        """
        samxz = np.abs(sam)
        samyz = np.sign(sam)
        
        sx, _ = InfoHDP.Sint(samxz)
        sy, _ = InfoHDP.Sint(samyz)
        sxy, _ = InfoHDP.Sint(sam)
        
        insb = sx + sy - sxy
        return insb, sx, sy, sxy

    @staticmethod
    def n10sam(sam: np.ndarray) -> List[List[int]]:
        """
        Compute n10 statistics from samples.
        
        Args:
            sam (np.ndarray): Sample data.
        
        Returns:
            List[List[int]]: n10 statistics.
        """
        samx = np.abs(sam)
        unique, counts = np.unique(samx, return_counts=True)
        n10 = [[np.sum(sam == x), np.sum(sam == -x)] for x in unique]
        return n10

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
            return -InfoHDP.logLb(np.exp(log_b), kx, n10, noprior)
        
        result = optimize.minimize_scalar(objective, bounds=(-10, 10), method='bounded')
        return np.exp(result.x)

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
        
