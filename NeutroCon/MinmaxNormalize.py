from Link import register_step
import numpy as np
from ._config import Debug
from typing import Literal


@register_step(Debug=Debug)
def MinmaxNormalize(context: np.ndarray, direction: Literal[0, 1] = 0) -> np.ndarray:
    """
    Normalizes the matrix using min-max normalization to scale values between 0 and 1.

    Parameters:
    - context (np.ndarray): The input 2D matrix to normalize.
    - direction (int): The direction along which to normalize.

    Returns:
    - np.ndarray: The normalized matrix with values scaled between 0 and 1.
    """
    if direction not in [0, 1]:
        raise ValueError("Direction must be either 0 or 1")

    min_value = np.min(context, axis=direction, keepdims=True)
    max_value = np.max(context, axis=direction, keepdims=True)

    range_value = max_value - min_value
    if np.any(range_value == 0):
        return np.zeros_like(context)

    return (context - min_value) / range_value
