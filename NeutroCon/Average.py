from typing import Dict, List
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def Average(context: Dict[str, List[float]]) -> List[float]:
    lists = context.values()
    max_length = max(len(lst) for lst in lists)
    sums = [0] * max_length
    counts = [0] * max_length
    for lst in lists:
        for index, value in enumerate(lst):
            sums[index] += value
            counts[index] += 1
    return [sums[index] / counts[index] if counts[index] > 0 else 0.0 for index in range(max_length)]
