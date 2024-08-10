from Link import register_step
from ._config import Debug
import numpy as np


@register_step(Debug=Debug)
def Orthonormalize(context: np.ndarray) -> np.ndarray:
    """
    Perform Gram-Schmidt orthonormalization on the columns of matrix.

    Args:
    context (np.ndarray): Matrix with columns representing vectors (m x n).

    Returns:
    np.ndarray: Matrix with orthonormal columns (m x n).
    """
    m, n = context.shape
    Q = np.zeros((m, n))
    for j in range(n):
        v = context[:, j]
        for i in range(j):
            q = Q[:, i]
            v -= np.dot(v, q) * q
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            Q[:, j] = v / norm
        else:
            Q[:, j] = v
    return Q
