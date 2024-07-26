# infohdp/generators/sample.py

import numpy as np
from typing import List, Tuple, Union
from ..constants import DEFAULT_NUM_STATES, DEFAULT_NUM_SAMPLES

def gen_samples_prior(qij: np.ndarray, M: int = DEFAULT_NUM_SAMPLES, Ns: int = DEFAULT_NUM_STATES) -> np.ndarray:
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

def gen_samples_prior_t(pi: np.ndarray, pjdadoi: np.ndarray, M: int = DEFAULT_NUM_SAMPLES, Ns: int = DEFAULT_NUM_STATES) -> List[Tuple[int, int]]:
    """
    Generates samples {state x, state y} given the probabilities.

    Args:
        pi (np.ndarray): Probability distribution for X.
        pjdadoi (np.ndarray): Conditional probability distribution for Y given X.
        M (int): Number of samples to generate.
        Ns (int): Number of states.

    Returns:
        List[Tuple[int, int]]: List of tuples, each containing a sample (x, y).
    """
    Ny = pjdadoi.shape[1]
    samx = np.random.choice(range(Ns), size=M, p=np.abs(pi))
    sam = [(x, np.random.choice(range(Ny), p=np.abs(pjdadoi[x]))) for x in samx]
    return sam
