import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def Project(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Project x array on y array by numpy.dot function

    Parameters:
    - x (numpy.ndarray): Array 1
    - y (numpy.ndarray): Array 2

    Returns:
    - projected_matrix (numpy.ndarray): Matrix resulting from numpy.dot function
    """
    return np.dot(x, y)
