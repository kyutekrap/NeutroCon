import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def WeightedAverage(context: np.ndarray, direction: int = 0) -> np.ndarray:
    """
    Calculate the weighted average of a 2D NumPy array along a specified axis.
    The weights increase linearly from 1 along the specified axis.

    Args:
    context (np.ndarray): 2D NumPy array.
    direction (int): The axis along which to calculate the weighted average.

    Returns:
    np.ndarray: The weighted average along the specified axis.
    """
    if context.ndim != 2:
        raise ValueError("The input array must be 2D.")

    if direction not in (0, 1):
        raise ValueError("Direction must be 0 or 1.")

    size = context.shape[direction]
    weights = np.arange(1, size + 1)
    weighted_sum = np.sum(context * weights, axis=direction)
    total_weight = np.sum(weights)
    return weighted_sum / total_weight
