import numpy as np
from typing import Union, List, Tuple

def strue(p: np.ndarray) -> float:
    """
    Calculate the true entropy of a probability distribution.

    Args:
        p (np.ndarray): Probability distribution.

    Returns:
        float: Entropy value.
    """
    p = np.asarray(p)
    return -np.sum(p * np.log(p, where=(p > 0), out=np.zeros_like(p)))

def sxtrue(p: np.ndarray) -> float:
    """
    Calculate the true entropy of X.

    Args:
        p (np.ndarray): Joint probability distribution P(X,Y).

    Returns:
        float: Entropy of X.
    """
    p_reshaped = p.reshape(-1, 2)
    return strue(np.sum(p_reshaped, axis=1))

def sytrue(p: np.ndarray) -> float:
    """
    Calculate the true entropy of Y.

    Args:
        p (np.ndarray): Joint probability distribution P(X,Y).

    Returns:
        float: Entropy of Y.
    """
    p_reshaped = p.reshape(-1, 2)
    return strue(np.sum(p_reshaped, axis=0))

def itrue(p: np.ndarray) -> float:
    """
    Calculate the true mutual information.

    Args:
        p (np.ndarray): Joint probability distribution P(X,Y).

    Returns:
        float: Mutual information value.
    """
    return -strue(p) + sytrue(p) + sxtrue(p)

# Add any other core functions that are fundamental to your package
