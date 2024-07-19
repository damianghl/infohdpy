import numpy as np
from scipy import stats, special
from scipy.optimize import minimize
from scipy import optimize, integrate
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

    @classmethod
    def Smap(cls, sam: np.ndarray) -> float:
        """
        Calculates the maximum a posteriori entropy estimate (S_MAP).

        Args:
            sam (np.ndarray): Sample data.

        Returns:
            float: Maximum a posteriori entropy estimate.
        """
        nn = len(sam)
        dkmz = cls.dkm2(sam)
        kz = len(np.unique(sam))
        az = cls.asol(nn, kz)
        smap = cls.Spost(az, nn, dkmz)
        return smap

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
        az = InfoHDP.asol(nn, kz) # checked
        
        log_az = np.log(az)
        lower_bound = log_az - 3  # Equivalent to log(az/10)
        upper_bound = log_az + 3  # Equivalent to log(az*10)

        def integrand_normalization(log_x):
            return np.exp(InfoHDP.logLa(np.exp(log_x), nn, kz))

        def integrand_weighted_spost(log_x):
            x = np.exp(log_x)
            return InfoHDP.Spost(x, nn, dkmz) * np.exp(InfoHDP.logLa(x, nn, kz))

        def integrand_weighted_spost2(log_x):
            x = np.exp(log_x)
            return (InfoHDP.Spost(x, nn, dkmz)**2) * np.exp(InfoHDP.logLa(x, nn, kz))

        # Calculate normalization constant
        norm_const, norm_error = integrate.quad(integrand_normalization, lower_bound, upper_bound)

        # Calculate weighted integral of Spost
        weighted_integral, weighted_error = integrate.quad(integrand_weighted_spost, lower_bound, upper_bound)

        # Calculate weighted integral of Spost
        weighted_integral2, weighted_error2 = integrate.quad(integrand_weighted_spost2, lower_bound, upper_bound)

        # Normalize the result
        sint = weighted_integral / norm_const
        sint2 = weighted_integral2 / norm_const

        # Estimate error (this is an approximation and may need refinement)
        dsint = np.sqrt(sint2 - sint**2)

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
        
    @staticmethod
    def gen_nasty_pij(alfa, psure, type=1, ndist=1, Ns=10000):
        """
        Generates probabilities pij such that q_i ~ DP(alfa_Ns) and q_j|i = {psure(Prob=0.25), 0.5(Prob=0.50), 1.-psure(Prob=0.25)}..
        Args:
            alfa (float): Concentration parameter for Dirichlet process.
            type (int): different ways to choose q_j|i.
            ndist (int): Number of distributions to generate.
            Ns (int): Number of states.
        
        Returns:
            np.ndarray: Probability distributions.
        """
        alist = np.full(Ns, alfa / Ns)
        if type == 1:
            prdel = [0.25, 0.5, 0.25]
        elif type == 2:
            prdel = [1/3, 1/3, 1/3]
        else:
            raise ValueError("Invalid type")
        
        bes = np.random.choice([psure, 0.5, 1 - psure], size=Ns, p=prdel)
        pi = np.random.dirichlet(alist, size=ndist)
        pi = np.column_stack((pi, 1 - np.sum(pi, axis=1)))
        
        pij = np.zeros((ndist, 2 * Ns))
        for k in range(ndist):
            pij[k, ::2] = pi[k] * bes
            pij[k, 1::2] = pi[k] * (1 - bes)
        
        return pij
    
    @staticmethod
    def gen_nasty_pij2(psure, type=2, ndist=1, Ns=10000):
        """
        Generates probabilities pij such that q_i ~ 1/Ns and q_j|i = {psure(Prob=0.25), 0.5(Prob=0.50), 1.-psure(Prob=0.25)}..
        Args:
            alfa (float): Concentration parameter for Dirichlet process.
            type (int): different ways to choose q_j|i.
            ndist (int): Number of distributions to generate.
            Ns (int): Number of states.
        
        Returns:
            np.ndarray: Probability distributions.
        """
        if type == 1:
            prdel = [0.25, 0.5, 0.25]
        elif type == 2:
            prdel = [1/3, 1/3, 1/3]
        else:
            raise ValueError("Invalid type")
        
        bes = np.random.choice([psure, 0.5, 1 - psure], size=Ns, p=prdel)
        pi = np.full((ndist, Ns), 1 / Ns)
        
        pij = np.zeros((ndist, 2 * Ns))
        for k in range(ndist):
            pij[k, ::2] = pi[k] * bes
            pij[k, 1::2] = pi[k] * (1 - bes)
        
        return pij
    
    @staticmethod
    def D2expalogL(ex, n, k):
        """
        Calculates the second derivative of log-likelihood of alpha with respect to log(alpha).

        This function is used to determine intervals for integration in the estimation process.

        Args:
            ex (float): Exponential of alpha value (exp(alpha)).
            n (int): Total number of samples.
            k (int): Number of unique samples.

        Returns:
            float: Second derivative of log-likelihood with respect to log(alpha).
        """
        return np.exp(ex) * (special.digamma(1 + np.exp(ex)) - special.digamma(np.exp(ex) + n) + 
                            np.exp(ex) * (special.polygamma(1, 1 + np.exp(ex)) - special.polygamma(1, np.exp(ex) + n)))

    @staticmethod
    def intEa(xx, nz, kz, nsig=3):
        """
        Calculates the interval for integration in log(alpha).

        This function determines the range over which to integrate when estimating alpha.

        Args:
            xx (float): Alpha value.
            nz (int): Total number of samples.
            kz (int): Number of unique samples.
            nsig (float, optional): Number of standard deviations to use for the interval. Defaults to 3.

        Returns:
            Tuple[float, float]: Lower and upper bounds of the integration interval in log(alpha).
        """
        sigea = np.sqrt(-InfoHDP.D2expalogL(np.log(xx), nz, kz))
        ead = np.log(xx) - nsig * sigea
        eau = np.log(xx) + nsig * sigea
        return ead, eau
    
    @classmethod
    def InsbCon(cls, sam):
        """
        Calculates the NSB (Nemenman-Shafee-Bialek) estimate for mutual information I = S(X) - S(X|Y).

        This method provides an alternative way to estimate mutual information compared to the standard NSB approach.

        Args:
            sam (np.ndarray): Sample data.

        Returns:
            Tuple[float, float, float, float]: Estimated mutual information, S(X), S(X|Y=1), and S(X|Y=0).
        """
        nn = len(sam)
        samxz = np.abs(sam)
        samx1z = sam[sam > 0]
        samx0z = sam[sam < 0]
        nn1 = len(samx1z)
        nn0 = len(samx0z)
        
        sx = cls.Sint(samxz)[0]
        sx1 = cls.Sint(samx1z)[0]
        sx0 = cls.Sint(samx0z)[0]
        
        insb = sx - (sx1 * nn1 + sx0 * nn0) / nn
        return insb, sx, sx1, sx0
    
    @staticmethod
    def bsolE(kx, n10):
        """
        Solves for the hyperparameter beta by searching in log(beta) space.

        This method finds the optimal beta value that maximizes the likelihood of the observed data.

        Args:
            kx (int): Number of unique X samples.
            n10 (np.ndarray): Array of counts for each state, where n10[i] = [n_i1, n_i0].

        Returns:
            float: Optimal beta value.
        """
        def objective(x):
            exp_x = np.exp(x)
            return 1 + exp_x * (2 * kx * (special.digamma(2 * exp_x) - special.digamma(exp_x)) +
                                (4 * special.polygamma(2, 2 * exp_x + 1) - special.polygamma(2, exp_x + 1)) /
                                (2 * special.polygamma(1, 2 * exp_x + 1) - special.polygamma(1, exp_x + 1)) +
                                np.sum(special.digamma(exp_x + n10[:, 0]) + special.digamma(exp_x + n10[:, 1]) -
                                    2 * special.digamma(2 * exp_x + n10[:, 0] + n10[:, 1])))
        
        result = optimize.root_scalar(objective, x0=1, method='newton')
        return np.exp(result.root)
    
    @classmethod
    def IhdpMAP(cls, sam, onlyb=0, noprior=0):
        """
        Calculates the MAP (Maximum A Posteriori) estimate of mutual information using InfoHDP.

        This method provides an estimate of mutual information based on the InfoHDP approach.

        Args:
            sam (np.ndarray): Sample data.
            onlyb (int, optional): If 1, uses only beta (no alpha, i.e., no pseudocounts). Defaults to 0.
            noprior (int, optional): If 1, no prior is used for beta. Defaults to 0.

        Returns:
            float: Estimated mutual information.
        """
        nn = len(sam)
        a1 = 0
        
        if onlyb != 1:
            kk = len(np.unique(sam))
            a1 = cls.asol(nn, kk)
        
        samx = np.abs(sam)
        kx = len(np.unique(samx))
        n10 = cls.n10sam(sam)
        b1 = cls.bsol(kx, n10, noprior)
        
        sy = cls.smaxlik(np.sign(sam))
        sycx = cls.SYconX(a1, b1, nn, n10)
        
        ihdp = sy - sycx
        return ihdp
    
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
    
    @classmethod
    def intEb(cls, bx, kx, n10, nsig=3, noprior=0):
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
        sigeb = np.sqrt(-cls.D2expblogL(np.log(bx), kx, n10, noprior))
        ebd = np.log(bx) - nsig * sigeb
        ebu = np.log(bx) + nsig * sigeb
        return ebd, ebu
    
    @classmethod
    def IhdpIntb(cls, sam, onlyb=0, noprior=0):
        """
        Calculates the InfoHDP estimator by integrating over the peak of the posterior (only in beta).

        Args:
            sam (np.ndarray): Sample data.
            onlyb (int, optional): If 1, only uses beta (no alpha). Defaults to 0.
            noprior (int, optional): If 1, no prior is used. Defaults to 0.

        Returns:
            Tuple[float, float, float]: Estimated mutual information, standard deviation of the estimate, and estimated conditional entropy S(Y|X).
        """
        nn = len(sam)
        az = 0
        
        if onlyb != 1:
            kk = len(np.unique(sam))
            az = cls.asol(nn, kk)
        
        samx = np.abs(sam)
        kx = len(np.unique(samx))
        n10 = cls.n10sam(sam)
        bz = cls.bsol(kx, n10, noprior)
        sy = cls.smaxlik(np.sign(sam))
        
        logLbz = cls.logLb(bz, kx, n10, noprior)
        ebd, ebu = cls.intEb(bz, kx, n10, 3, noprior)
        listEb = np.linspace(ebd, ebu, 25)
        listLogL = np.exp(cls.logLb(np.exp(listEb), kx, n10, noprior) - logLbz)
        listLogL /= np.sum(listLogL)
        
        sint = np.sum([cls.SYconX(az, np.exp(eb), nn, n10) * ll for eb, ll in zip(listEb, listLogL)])
        s2int = np.sum([cls.SYconX2(np.exp(eb), nn, n10) * ll for eb, ll in zip(listEb, listLogL)])
        dsint = np.sqrt(s2int - sint**2)
        
        ihdp = sy - sint
        return ihdp, dsint, sint
    
    @staticmethod
    def genPriorPijT(alfa, beta, qy, Ns=10000):
        """
        Generates probabilities {pi, pj|i, pij} with prior and marginal qy.

        Args:
            alfa (float): Concentration parameter for Dirichlet process.
            beta (float): Parameter for Beta distribution.
            qy (List[float]): Marginal distribution for Y.
            Ns (int, optional): Number of states. Defaults to 10000.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing pi, pj|i, and pij.
        """
        alist = np.full(Ns, alfa / Ns)
        pjdadoi = np.random.dirichlet(beta * qy, size=Ns)
        pjdadoi = np.column_stack((pjdadoi, 1 - np.sum(pjdadoi, axis=1)))
        pi = np.random.dirichlet(alist)
        pi = np.append(pi, 1 - np.sum(pi))
        pij = np.outer(pi, pjdadoi.T).T
        return pi, pjdadoi, pij

    @staticmethod
    def genSamplesPriorT(pi, pjdadoi, M, Ns=10000):
        """
        Generates samples {state x, state y} given the probabilities.

        Args:
            pi (np.ndarray): Probability distribution for X.
            pjdadoi (np.ndarray): Conditional probability distribution for Y given X.
            M (int): Number of samples to generate.
            Ns (int, optional): Number of states. Defaults to 10000.

        Returns:
            List[Tuple[int, int]]: List of tuples, each containing a sample (x, y).
        """
        Ny = pjdadoi.shape[1]
        samx = np.random.choice(range(Ns), size=M, p=np.abs(pi))
        sam = [(x, np.random.choice(range(Ny), p=np.abs(pjdadoi[x]))) for x in samx]
        return sam
    
    @staticmethod
    def nxysam(sam, Ny):
        """
        Counts for each x that occurs, the number of y samples (result in matrix Kx x Ny).

        Args:
            sam (List[Tuple[int, int]]): List of samples, where each sample is a tuple (x, y).
            Ny (int): Number of possible y values.

        Returns:
            np.ndarray: 2D numpy array of counts.
        """
        samx = [s[0] for s in sam]
        tsamx = np.unique(samx)
        nxy = [[sum(1 for s in sam if s[0] == x and s[1] == y) for y in range(Ny)] for x in tsamx]
        return np.array(nxy)
    
    @staticmethod
    def logLbT(b, qy, nxy):
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
        ll += np.sum(np.sum(special.gammaln(1 + b * qy + nxy), axis=1) - special.gammaln(b + np.sum(nxy, axis=1)))
        return ll
    
    @classmethod
    def bsolT(cls, qy, nxy):
        """
        Gives the beta that maximizes the marginal log-likelihood, given an estimated qy and counts.

        Args:
            qy (np.ndarray): Marginal distribution for Y.
            nxy (np.ndarray): Count matrix from nxysam.

        Returns:
            float: Optimal beta value.
        """
        def objective(ebb):
            return -cls.logLbT(np.exp(ebb), qy, nxy)
        
        result = optimize.minimize_scalar(objective, method='brent')
        return np.exp(result.x)
    
    @staticmethod
    def SYconXT(bb, nn, qy, nxy):
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
    
    @classmethod
    def IhdpMAPT(cls, sam, ML=0):
        """
        Gives the MAP estimate of mutual information using InfoHDP.

        Args:
            sam (List[Tuple[int, int]]): List of samples, where each sample is a tuple (x, y).
            ML (int, optional): If 1, use Maximum Likelihood estimation for qy; if 0, use posterior mean. Defaults to 0.

        Returns:
            float: Estimated mutual information.
        """
        nn = len(sam)
        ny = max(s[1] for s in sam) + 1
        nxy = cls.nxysam(sam, ny)
        
        if ML == 1:
            qye = np.sum(nxy, axis=0) / np.sum(nxy)
        else:
            qye = (np.sum(nxy, axis=0) + 1/ny) / (np.sum(nxy) + 1)
        
        b1 = cls.bsolT(qye, nxy)
        sy = cls.strue(qye)
        sycx = cls.SYconXT(b1, nn, qye, nxy)
        ihdp = sy - sycx
        return ihdp
    
    @classmethod
    def InaiveT(cls, sam):
        """
        Gets the ML estimate for I(X;Y), assuming samples as a list of (x, y) tuples.

        Args:
            sam (List[Tuple[int, int]]): List of samples, where each sample is a tuple (x, y).

        Returns:
            float: ML estimate of mutual information.
        """
        nn = len(sam)
        samxz = [s[0] for s in sam]
        samyz = [s[1] for s in sam]
        
        dkmz = cls.dkm2(sam)
        dkmzX = cls.dkm2(samxz)
        dkmzY = cls.dkm2(samyz)
        
        Iml = cls.snaive(nn, dkmzX) + cls.snaive(nn, dkmzY) - cls.snaive(nn, dkmz)
        return Iml
    
    @classmethod
    def InsbT(cls, sam):
        """
        Gives NSB estimate for I = S(X) + S(Y) - S(X,Y), assuming samples as a list of (x, y) tuples.

        Args:
            sam (List[Tuple[int, int]]): List of samples, where each sample is a tuple (x, y).

        Returns:
            Tuple[float, float, float, float]: Tuple containing (I, S(X), S(Y), S(X,Y)).
        """
        nn = len(sam)
        samxz = [s[0] for s in sam]
        samyz = [s[1] for s in sam]
        
        sx, _ = cls.Sint(samxz)
        sy, _ = cls.Sint(samyz)
        sxy, _ = cls.Sint(sam)
        
        insb = sx + sy - sxy
        return insb, sx, sy, sxy
    
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
    
    @classmethod
    def varSYconX(cls, bb, nn, n10):
        """
        Calculates the variance of S(Y|X) for fixed beta.

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            n10 (List[List[int]]): n10 statistics.

        Returns:
            float: Variance of S(Y|X).
        """
        return (1 / (nn**2)) * sum((n1 + n0)**2 * cls.varSYx(bb, n0, n1) for n1, n0 in n10)

    @classmethod
    def SYconX2(cls, bb, nn, n10):
        """
        Calculates the second moment of S(Y|X) for fixed beta.

        Args:
            bb (float): Beta value.
            nn (int): Total number of samples.
            n10 (List[List[int]]): n10 statistics.

        Returns:
            float: Second moment of S(Y|X).
        """
        return cls.varSYconX(bb, nn, n10) + (cls.SYconX(0., bb, nn, n10))**2
    