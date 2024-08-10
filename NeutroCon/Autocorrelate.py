from Link import register_step
from ._config import Debug
import numpy as np


@register_step(Debug=Debug)
def Autocorrelate(context: np.ndarray) -> np.ndarray:
    """
    Compute the autocorrelation of a 2D matrix.

    Args:
    context (np.ndarray): Input 2D matrix.

    Returns:
    np.ndarray: Autocorrelation matrix.
    """
    rows, cols = context.shape
    matrix_mean = np.mean(context)
    autocorr = np.zeros((rows, cols))
    matrix_centered = context - matrix_mean
    for i in range(rows):
        for j in range(cols):
            shift_matrix = np.roll(matrix_centered, shift=(i, j), axis=(0, 1))
            autocorr[i, j] = np.sum(matrix_centered * shift_matrix)
    central_value = autocorr[0, 0]
    if central_value != 0:
        autocorr /= central_value
    return autocorr
