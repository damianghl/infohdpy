import numpy as np
from typing import List, Tuple

def freq_of_frequencies(sam: np.ndarray) -> List[Tuple[int, int]]:
    """
    Compute frequency of frequencies.

    Args:
        sam (np.ndarray): Sample data.

    Returns:
        List[Tuple[int, int]]: Frequency of frequencies.
    """
    
    unique, counts = np.unique(sam, return_counts=True, axis=0)
    unique_counts, count_counts = np.unique(counts, return_counts=True)
    return sorted(zip(unique_counts, count_counts))

def count_nxy_binary(sam: np.ndarray) -> List[List[int]]:
    """
    Counts for each x that occurs, the number of y=+-1 samples.

    Args:
        sam (np.ndarray): Sample data.

    Returns:
        List[List[int]]: Counts of y=+-1.
    """
    samx = np.abs(sam)
    unique, counts = np.unique(samx, return_counts=True)
    n10 = [[np.sum(sam == x), np.sum(sam == -x)] for x in unique]
    return n10

def count_nxy_multiclass(sam: List[Tuple[int, int]]) -> np.ndarray:
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
    samy = [s[1] for s in sam]
    tsamy = np.unique(samy)
    nxy = [[sum(1 for s in sam if s[0] == x and s[1] == y) for y in tsamy] for x in tsamx]
    return np.array(nxy)
