from typing import Dict, List
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def Subtract(context: Dict[str, List[float]]) -> List[float]:
    lists = list(context.values())
    result = lists[0][:]
    for lst in lists[1:]:
        for index in range(min(len(result), len(lst))):
            result[index] -= lst[index]
    for i in range(len(result)):
        if i >= len(lists[1:]):
            result[i] -= 0
    return result
