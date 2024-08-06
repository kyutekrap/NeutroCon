from typing import List
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def RollingDifference(lst: List[float], period: int = 1) -> List[float]:
    return [lst[i + period] - lst[i] for i in range(len(lst) - period)]
