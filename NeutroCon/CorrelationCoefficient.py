import numpy as np
from Link import register_step
from ._config import Debug
from typing import Literal


@register_step(Debug=Debug)
def CorrelationCoefficient(context: np.ndarray, direction: Literal[0, 1] = 0) -> np.ndarray:
    """
    Compute the correlation coefficient matrix of a 2D matrix.

    Args:
    context (np.ndarray): Input 2D matrix.

    Returns:
    np.ndarray: Correlation coefficient matrix.
    """
    return np.corrcoef(context, rowvar=False if direction == 0 else True)
