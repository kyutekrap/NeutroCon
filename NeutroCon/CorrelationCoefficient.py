from typing import List
from math import sqrt
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def CorrelationCoefficient(x: List[float], y: List[float], period: int = None) -> List[float]:
    if len(x) != len(y):
        raise ValueError("The length of x and y must be the same.")

    n = len(x)

    if period is None:
        period = n

    if period > n:
        raise ValueError("Period cannot be greater than the length of the lists.")
    if period < 2:
        raise ValueError("Period must be at least 2 to calculate correlation.")

    def calculate_correlation(start: int, end: int) -> float:
        sub_x = x[start:end]
        sub_y = y[start:end]
        k = end - start

        sum_x = sum(sub_x)
        sum_y = sum(sub_y)
        sum_x2 = sum(val ** 2 for val in sub_x)
        sum_y2 = sum(val ** 2 for val in sub_y)
        sum_xy = sum(xi * yi for xi, yi in zip(sub_x, sub_y))

        numerator = k * sum_xy - sum_x * sum_y
        denominator = sqrt((k * sum_x2 - sum_x ** 2) * (k * sum_y2 - sum_y ** 2))

        if denominator == 0:
            return float('nan')

        return numerator / denominator

    results = []
    for i in range(n - period + 1):
        correlation = calculate_correlation(i, i + period)
        results.append(correlation)

    return results
