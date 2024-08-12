from Link import register_step
from ._config import Debug
import numpy as np


@register_step(Debug=Debug)
def LeastSquare(context: np.ndarray) -> np.ndarray:
    """
    Compute the least squares (squared values) of each element in a NumPy array.

    Args:
    context (np.ndarray): 2D NumPy array.

    Returns:
    np.ndarray: A NumPy array with each element squared.
    """
    return np.square(context)
