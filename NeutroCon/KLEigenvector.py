from Link import register_step
import numpy as np
from ._config import Debug


@register_step(Debug=Debug)
def KLEigenvector(context: np.ndarray) -> np.ndarray:
    """
    Compute the basis of a matrix using the Karhunen-Loève (KL) expansion.

    Parameters:
    context (np.ndarray): 2D NumPy array.

    Returns:
    np.ndarray: The basis vectors (principal components) of the matrix.
    """
    centered = context - np.mean(context, axis=0)
    covariance_matrix = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    return eigenvectors[:, sorted_indices]
