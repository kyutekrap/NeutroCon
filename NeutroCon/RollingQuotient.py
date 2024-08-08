import numpy as np
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def RollingQuotient(arr: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Compute the rolling quotient of a NumPy array.

    Args:
    arr (np.ndarray): 1D NumPy array.
    period (int): The period over which to calculate the quotient.

    Returns:
    np.ndarray: Array of rolling quotients.
    """
    if period < 1:
        raise ValueError("Period must be at least 1.")
    result = np.empty(len(arr) - period + 1, dtype=np.float64)
    for i in range(len(result)):
        segment = arr[i:i + period]
        if np.any(segment[1:] == 0):
            result[i] = np.nan
        else:
            result[i] = segment[0] / np.prod(segment[1:])
    return result
