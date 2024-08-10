import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def Normalize(context: np.ndarray) -> np.ndarray:
    """
    Normalize a 2d array into unit vector.

    Args:
    context (np.ndarray): A 2D NumPy array.

    Returns:
    np.ndarray: Resulting array after normalization.
    """
    norm = np.linalg.norm(context)
    return context / norm if norm > 0 else context
