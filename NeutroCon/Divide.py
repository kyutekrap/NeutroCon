from Link import register_step
from config import Debug
import numpy as np
from typing import Literal


@register_step(Debug=Debug)
def Divide(context: np.array, direction: Literal[0, 1] = 0) -> np.array:
    """
    Divide elements along the specified axis.

    Args:
    context (np.array): A 2D NumPy array.
    direction (Literal[0, 1]): Axis along which to perform the division.

    Returns:
    np.array: Resulting array after division.
    """
    if direction == 0:
        result = context[0, :]
        for i in range(1, context.shape[0]):
            result /= context[i, :]
    elif direction == 1:
        result = context[:, 0]
        for i in range(1, context.shape[1]):
            result /= context[:, i]
    else:
        raise ValueError("Direction must be 0 (columns) or 1 (rows).")
    return result
