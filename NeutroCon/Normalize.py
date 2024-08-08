import numpy as np
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def Normalize(context: np.array) -> np.array:
    """
    Normalize a 2d array into unit vector

    Args:
    context (np.array): A 2D NumPy array.

    Returns:
    np.array: Resulting array after normalization.
    """
    norm = np.linalg.norm(context)
    return context / norm if norm > 0 else context
