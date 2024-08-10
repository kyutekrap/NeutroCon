from Link import register_step
from ._config import Debug
import numpy as np


@register_step(Debug=Debug)
def Inverse(context: np.ndarray) -> np.ndarray:
    """
    Get the inverse using the Gauss-Jordan elimination method.

    Args:
    context (np.array): A 2D NumPy array.

    Returns:
    np.array: The inverse of the matrix.

    Raises:
    ValueError: If the matrix is not square or is singular (non-invertible).
    """
    n, m = context.shape
    if n != m:
        raise ValueError("The matrix must be square to find its inverse.")
    augmented_matrix = np.hstack((context, np.eye(n)))
    for i in range(n):
        if augmented_matrix[i, i] == 0:
            for k in range(i + 1, n):
                if augmented_matrix[k, i] != 0:
                    augmented_matrix[[i, k]] = augmented_matrix[[k, i]]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")
        augmented_matrix[i] /= augmented_matrix[i, i]
        for j in range(n):
            if i != j:
                augmented_matrix[j] -= augmented_matrix[j, i] * augmented_matrix[i]
    return augmented_matrix[:, n:]
