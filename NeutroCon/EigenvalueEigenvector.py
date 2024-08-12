from Link import register_step
import numpy as np
from ._config import Debug


@register_step(Debug=Debug)
def EigenvalueEigenvector(context: np.ndarray) -> tuple:
    """
    Compute the eigenvalues and eigenvector of a square matrix.

    Parameters:
    context (np.ndarray): A square matrix (2D numpy array).

    Returns:
    np.ndarray: Eigenvalues and eigenvector of the matrix.
    """
    if context.shape[0] != context.shape[1]:
        raise ValueError("The matrix must be square.")
    return np.linalg.eig(context)
