import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def RollingDifference(arr: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Compute the rolling difference of a NumPy array.

    Args:
    arr (np.ndarray): 1D NumPy array.
    period (int): The period over which to calculate the difference.

    Returns:
    np.ndarray: Array of rolling differences.
    """
    if period < 1:
        raise ValueError("Period must be at least 1.")

    return arr[period-1:] - arr[:-period+1]
