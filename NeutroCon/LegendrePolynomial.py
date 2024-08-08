from Link import register_step
from config import Debug
import numpy as np


@register_step(Debug=Debug)
def LegendrePolynomial(context: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Compute Legendre polynomials for a given order element-wise.

    Args:
    context (np.ndarray): Input values.
    order (int): Order of the Legendre polynomials.

    Returns:
    np.ndarray: Array of Legendre polynomial values.
    """
    def compute_legendre(x, order):
        """
        Define the Legendre polynomial computation using recurrence relation
        """
        P = np.zeros(order + 1)
        P[0] = 1
        if order > 0:
            P[1] = x
        for n in range(1, order):
            P[n + 1] = ((2 * n + 1) * x * P[n] - n * P[n - 1]) / (n + 1)
        return P[order]
    vectorized_legendre = np.vectorize(lambda x_val: compute_legendre(x_val, order))
    return vectorized_legendre(context)
