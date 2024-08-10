import numpy as np
from Link import register_step
from ._config import Debug
from typing import Literal


@register_step(Debug=Debug)
def Covariance(context: np.ndarray, direction: Literal[0, 1] = 0) -> np.ndarray:
    """
    Compute the covariance matrix of a 2D matrix.

    Args:
    context (np.ndarray): Input 2D matrix.
    direction (Literal[0, 1]): Axis.

    Returns:
    np.ndarray: Covariance matrix of the input matrix.
    """
    return np.cov(context, rowvar=False if direction == 0 else True)
