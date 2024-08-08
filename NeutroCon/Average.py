from Link import register_step
from config import Debug
import numpy as np
from typing import Literal


@register_step(Debug=Debug)
def Average(context: np.array, direction: Literal[0, 1] = 0) -> np.array:
    """
    Calculate the average in a 2D NumPy array.

    Args:
    context (np.array): A 2D NumPy array.
    direction (Literal[0, 1]): Axis along which to perform the average.

    Returns:
    np.array: Resulting array after averaging.
    """
    return np.mean(context, axis=direction)
