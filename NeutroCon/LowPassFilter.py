from Link import register_step
import numpy as np
from ._config import Debug


@register_step(Debug=Debug)
def LowPassFilter(context: np.ndarray, filter_size: int = 3) -> np.ndarray:
    """
    Applies a low-pass filter to a 2D matrix using a box filter.

    Parameters:
    - context (np.ndarray): The input 2D matrix to filter.
    - filter_size (int): The size of the box filter (must be an odd number).

    Returns:
    - np.ndarray: The filtered 2D matrix.
    """
    if filter_size % 2 == 0:
        raise ValueError("filter_size must be an odd number.")

    rows, cols = context.shape

    box_filter = np.ones((filter_size, filter_size)) / (filter_size * filter_size)

    pad_width = filter_size // 2

    padded_matrix = np.pad(context, pad_width, mode='constant', constant_values=0)

    filtered_matrix = np.zeros_like(context)

    for i in range(rows):
        for j in range(cols):
            region = padded_matrix[i:i + filter_size, j:j + filter_size]
            filtered_matrix[i, j] = np.sum(region * box_filter)

    return filtered_matrix
