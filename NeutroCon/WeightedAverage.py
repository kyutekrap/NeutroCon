import numpy as np
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def WeightedAverage(arr: np.ndarray) -> float:
    """
    Calculate the weighted average of a NumPy array where weights increase linearly from 1.

    Args:
    arr (np.ndarray): 1D NumPy array.

    Returns:
    float: The weighted average of the array.
    """
    n = arr.size
    weights = np.arange(1, n + 1)
    weighted_sum = np.sum(arr * weights)
    total_weight = np.sum(weights)
    return weighted_sum / total_weight
