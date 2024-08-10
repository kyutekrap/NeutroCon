from Link import register_step
from ._config import Debug
import numpy as np


@register_step(Debug=Debug)
def HermitePolynomial(context: np.ndarray, order: int = 2) -> np.ndarray:
    """
    Compute Hermite polynomials for a given order element-wise.

    Args:
    context (np.ndarray): A 2D NumPy array.
    order (int): Order of the Hermite polynomials.

    Returns:
    np.ndarray: Array of Hermite polynomial values.
    """
    def compute_hermite(x, order):
        """
        Define the Hermite polynomial computation using recurrence relation
        """
        H = np.zeros(order + 1)
        H[0] = 1
        if order > 0:
            H[1] = 2 * x
        for n in range(1, order):
            H[n + 1] = 2 * x * H[n] - 2 * n * H[n - 1]
        return H[order]
    vectorized_hermite = np.vectorize(lambda x_val: compute_hermite(x_val, order))
    return vectorized_hermite(context)
