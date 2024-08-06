from typing import List
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def WeightedAverage(lst: List[float]) -> float:
    n = len(lst)
    weights = list(range(1, n + 1))
    weighted_sum = sum(value * weight for value, weight in zip(lst, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight
