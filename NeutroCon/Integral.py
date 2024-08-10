from Link import register_step
import numpy as np
from ._config import Debug
from typing import Literal


def trapezoidal_integral(values: np.ndarray, dx: int = 1) -> np.ndarray:
    """
    Compute the trapezoidal integral of a 1D array.

    Parameters:
    values (numpy.ndarray): 1D subarray of context.
    dx (int): The spacing between points (default is 1).

    Returns:
    np.ndarray: The integral computed using the trapezoidal rule.
    """
    if len(values) < 2:
        raise ValueError("At least two points are required for trapezoidal integration.")
    integral = 0.5 * (values[0] + values[-1])
    integral += np.sum(values[1:-1])
    integral *= dx
    return integral


def gradient_integral(values: np.ndarray, dx: int = 1) -> np.ndarray:
    """
    Compute the gradient integral of a 1D array.

    Parameters:
    values (numpy.ndarray): 1D subarray of context.
    dx (int): The spacing between points (default is 1).

    Returns:
    np.ndarray: The computed integral of the gradient.
    """
    gradient = np.gradient(values)
    return trapezoidal_integral(gradient)


@register_step(Debug=Debug)
def Integral(context: np.ndarray, direction: Literal[0, 1] = 0) -> np.ndarray:
    """
    Compute the trapezoidal integral with gradient.

    Parameters:
    context (np.ndarray): 2D array.
    direction (Literal[0, 1]): 0 to integrate by columns, 1 by rows.

    Returns:
    np.ndarray: 1D array with the integral for each row or column.
    """
    result = None
    num_rows, num_cols = context.shape
    if direction == 0:
        result = np.zeros(num_cols)
        for col in range(num_cols):
            subarray = context[:, col]
            result[col] = gradient_integral(subarray) + trapezoidal_integral(subarray)
    elif direction == 1:
        result = np.zeros(num_rows)
        for row in range(num_rows):
            subarray = context[row, :]
            result[row] = gradient_integral(subarray) + trapezoidal_integral(subarray)
    else:
        raise ValueError("Direction must be 0 or 1.")
    return result
