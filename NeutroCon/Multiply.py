from Link import register_step
from ._config import Debug
import numpy as np
from typing import Literal


@register_step(Debug=Debug)
def Multiply(context: np.ndarray, direction: Literal[0, 1] = 0) -> np.ndarray:
    """
    Multiply elements along the specified axis.

    Args:
    context (np.ndarray): A 2D NumPy array.
    direction (Literal[0, 1]): Axis along which to perform the multiplication.

    Returns:
    np.ndarray: Resulting array after multiplication.
    """
    return np.prod(context, axis=direction)
