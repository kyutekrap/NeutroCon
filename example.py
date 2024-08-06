from Link import *
from config import Debug
import logging
import random
import NeutroCon

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Create an array context
context = {
    'a': [random.uniform(1.0, 10.0) for _ in range(5)],
    'b': [random.uniform(1.0, 10.0) for _ in range(5)],
    'c': [random.uniform(1.0, 10.0) for _ in range(5)],
    'd': [random.uniform(1.0, 10.0) for _ in range(5)],
    'e': [random.uniform(1.0, 10.0) for _ in range(5)],
}


@register_flow(Debug=Debug)
def example_flow():
    CreateFlow([
        CreateStep.Add(context),
        Debugger.log(GetStep("Add")),
        CreateStep.Divide(context),
        Debugger.log(GetStep("Divide")),
        CreateStep.Multiply(context),
        Debugger.log(GetStep("Multiply")),
        CreateStep.Subtract(context),
        Debugger.log(GetStep("Subtract")),
        CreateStep.Average(context),
        Debugger.log(GetStep("Average")),
        CreateStep.CorrelationCoefficient(GetStep("Add"), GetStep("Subtract")),
        Debugger.log(GetStep("CorrelationCoefficient")),
        CreateStep.RollingDifference(GetStep("Multiply"), period=3),
        Debugger.log(GetStep("RollingDifference")),
        CreateStep.RollingSum(GetStep("Divide"), period=2),
        Debugger.log(GetStep("RollingSum")),
        CreateStep.WeightedAverage(GetStep("Subtract")),
        Debugger.log(GetStep("WeightedAverage"))
    ])


if __name__ == "__main__":
    example_flow()
