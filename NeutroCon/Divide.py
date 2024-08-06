from typing import Dict, List
from Link import register_step
from config import Debug


@register_step(Debug=Debug)
def Divide(context: Dict[str, List[float]]) -> List[float]:
    lists = list(context.values())
    result = lists[0][:]
    for lst in lists[1:]:
        for index in range(min(len(result), len(lst))):
            if lst[index] != 0:
                result[index] /= lst[index]
            else:
                result[index] = float('inf')
    return result
