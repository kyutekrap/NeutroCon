from Link import register_step
import numpy as np
from ._config import Debug


@register_step(Debug=Debug)
def GaussElimination(context: np.ndarray) -> tuple:
    """
    Applies Gaussian elimination to transform the matrix into upper triangular form.

    Parameters:
    - context (np.ndarray): The input 2D matrix to transform.

    Returns:
    - tuple
    np.ndarray: The transformed upper triangular matrix.
    list: Pivot positions of the transformed matrix.
    """
    matrix = context.astype(float)
    rows, cols = matrix.shape

    for i in range(min(rows, cols)):
        if matrix[i, i] == 0:
            for k in range(i + 1, rows):
                if matrix[k, i] != 0:
                    matrix[[i, k]] = matrix[[k, i]]
                    break

        matrix[i] = matrix[i] / matrix[i, i]

        for j in range(i + 1, rows):
            matrix[j] -= matrix[i] * matrix[j, i]

    pivot_positions = []
    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] != 0:
                pivot_positions.append((i, j))
                break

    return matrix, pivot_positions
