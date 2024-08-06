from typing import Dict, List
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def Add(context: Dict[str, List[float]]) -> List[float]:
    lists = list(context.values())
    max_length = max(len(lst) for lst in lists)
    sums = [0] * max_length
    for lst in lists:
        for index, value in enumerate(lst):
            sums[index] += value
    return sums
