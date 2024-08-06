from typing import Dict, List
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def Multiply(context: Dict[str, List[float]]) -> List[float]:
    lists = list(context.values())
    max_length = max(len(lst) for lst in lists)
    products = [1] * max_length
    for lst in lists:
        for index, value in enumerate(lst):
            products[index] *= value
    return products
