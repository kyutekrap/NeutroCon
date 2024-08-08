from Link import register_step
from config import Debug
import numpy as np
from typing import Literal


@register_step(Debug=Debug)
def Add(context: np.array, direction: Literal[0, 1] = 0) -> np.array:
    """
    Adds elements along the specified axis.

    Args:
    context (np.array): A 2D NumPy array.
    direction (Literal[0, 1]): Axis along which to perform the subtraction.

    Returns:
    np.array: Resulting array after addition.
    """
    return np.sum(context, axis=direction)
