import numpy as np
from typing import List, Union

def gen_prior_pij(alpha: float, beta: float, ndist: int = 1, Ns: int = 10000) -> np.ndarray:
    """
    Generate probability distributions using a Dirichlet for pi, and a Beta for pj|i.
    
    Args:
        alpha (float): Concentration parameter for Dirichlet process.
        beta (float): Parameter for Beta distribution.
        ndist (int): Number of distributions to generate.
        Ns (int): Number of states.
    
    Returns:
        np.ndarray: Probability distributions.
    """
    alist = np.full(Ns, alpha / Ns)
    bes = np.random.beta(beta, beta, size=Ns)
    pi = np.random.dirichlet(alist, size=ndist)
    pi = np.column_stack((pi, 1 - np.sum(pi, axis=1)))
    pij = np.array([np.concatenate([(pi[k, i] * bes[i], pi[k, i] * (1 - bes[i])) for i in range(Ns)]) for k in range(ndist)])
    return pij

def gen_nasty_pij(alfa: float, psure: float, type: int = 1, ndist: int = 1, Ns: int = 10000) -> np.ndarray:
    """
    Generates probabilities pij such that q_i ~ DP(alfa_Ns) and q_j|i = {psure(Prob=0.25), 0.5(Prob=0.50), 1.-psure(Prob=0.25)}.
    
    Args:
        alfa (float): Concentration parameter for Dirichlet process.
        psure (float): Probability for the 'sure' state.
        type (int): Different ways to choose q_j|i.
        ndist (int): Number of distributions to generate.
        Ns (int): Number of states.
    
    Returns:
        np.ndarray: Probability distributions.
    """
    alist = np.full(Ns, alfa / Ns)
    prdel = [0.25, 0.5, 0.25] if type == 1 else [1/3, 1/3, 1/3]
    
    bes = np.random.choice([psure, 0.5, 1 - psure], size=Ns, p=prdel)
    pi = np.random.dirichlet(alist, size=ndist)
    pi = np.column_stack((pi, 1 - np.sum(pi, axis=1)))
    
    pij = np.zeros((ndist, 2 * Ns))
    for k in range(ndist):
        pij[k, ::2] = pi[k] * bes
        pij[k, 1::2] = pi[k] * (1 - bes)
    
    return pij

def gen_nasty_pij2(psure: float, type: int = 2, ndist: int = 1, Ns: int = 10000) -> np.ndarray:
    """
    Generates probabilities pij such that q_i ~ 1/Ns and q_j|i = {psure(Prob=0.25), 0.5(Prob=0.50), 1.-psure(Prob=0.25)}.
    
    Args:
        psure (float): Probability for the 'sure' state.
        type (int): Different ways to choose q_j|i.
        ndist (int): Number of distributions to generate.
        Ns (int): Number of states.
    
    Returns:
        np.ndarray: Probability distributions.
    """
    prdel = [0.25, 0.5, 0.25] if type == 1 else [1/3, 1/3, 1/3]
    
    bes = np.random.choice([psure, 0.5, 1 - psure], size=Ns, p=prdel)
    pi = np.full((ndist, Ns), 1 / Ns)
    
    pij = np.zeros((ndist, 2 * Ns))
    for k in range(ndist):
        pij[k, ::2] = pi[k] * bes
        pij[k, 1::2] = pi[k] * (1 - bes)
    
    return pij

def gen_prior_pij_t(alfa: float, beta: float, qy: List[float], Ns: int = 10000) -> Union[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates probabilities {pi, pj|i, pij} with prior and marginal qy.

    Args:
        alfa (float): Concentration parameter for Dirichlet process.
        beta (float): Parameter for Beta distribution.
        qy (List[float]): Marginal distribution for Y.
        Ns (int): Number of states.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing pi, pj|i, and pij.
    """
    alist = np.full(Ns, alfa / Ns)
    pjdadoi = np.random.dirichlet(beta * np.array(qy), size=Ns)
    pi = np.random.dirichlet(alist)
    pi = np.append(pi, 1 - np.sum(pi))
    pij = np.outer(pi, pjdadoi.T).T
    return pi, pjdadoi, pij
