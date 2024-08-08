from Link import register_step
from config import Debug
import numpy as np


@register_step(Debug=Debug)
def LaguerrePolynomial(context: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Compute Laguerre polynomials for a given order element-wise.

    Args:
    context (np.ndarray): Input values.
    order (int): Order of the Laguerre polynomials.

    Returns:
    np.ndarray: Array of Laguerre polynomial values.
    """
    def compute_laguerre(x, order):
        """
        Define the Laguerre polynomial computation using recurrence relation
        """
        L = np.zeros(order + 1)
        L[0] = 1
        if order > 0:
            L[1] = 1 - x
        for n in range(1, order):
            L[n + 1] = ((2 * n + 1 - x) * L[n] - n * L[n - 1]) / (n + 1)
        return L[order]
    vectorized_laguerre = np.vectorize(lambda x_val: compute_laguerre(x_val, order))
    return vectorized_laguerre(context)
