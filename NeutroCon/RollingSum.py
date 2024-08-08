import numpy as np
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def RollingSum(arr: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Compute the rolling sum of a NumPy array.

    Args:
    arr (np.ndarray): 1D NumPy array.
    period (int): The period over which to calculate the sum.

    Returns:
    np.ndarray: Array of rolling sums.
    """
    if period < 1:
        raise ValueError("Period must be at least 1.")
    cumsum = np.cumsum(arr)
    return np.concatenate(([cumsum[period - 1]], cumsum[period:] - cumsum[:-period]))
