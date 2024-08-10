import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def RollingProduct(arr: np.ndarray, period: int = 1) -> np.ndarray:
    """
    Compute the rolling product of a NumPy array.

    Args:
    arr (np.ndarray): 1D NumPy array.
    period (int): The period over which to calculate the product.

    Returns:
    np.ndarray: Array of rolling products.
    """
    if period < 1:
        raise ValueError("Period must be at least 1.")
    result = np.empty(len(arr) - period + 1, dtype=np.float64)
    for i in range(len(result)):
        result[i] = np.prod(arr[i:i + period])
    return result
