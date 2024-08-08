from Link import *
from config import Debug
import logging
import NeutroCon
import numpy as np
from enum import IntEnum

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Create a matrix context
context = np.random.rand(3, 4)


class ColumnIndex(IntEnum):
    FIRST = 0
    SECOND = 1
    THIRD = 2


@register_flow(Debug=Debug)
def example_flow():
    CreateFlow([
        CreateStep.Add(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        Debugger.log(GetStep("Add")),
        # CreateStep.Divide(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Divide")),
        # CreateStep.Multiply(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Multiply")),
        # CreateStep.Subtract(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Subtract")),
        # CreateStep.Average(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Average")),
        # CreateStep.Normalize(context),
        # Debugger.log(GetStep("Normalize")),
        # CreateStep.WeightedAverage(context[:, ColumnIndex.FIRST]),
        # Debugger.log(GetStep("WeightedAverage")),
        # CreateStep.HermitePolynomial(context, order=2),
        # Debugger.log(GetStep("HermitePolynomial")),
        # CreateStep.LaguerrePolynomial(context, order=2),
        # Debugger.log(GetStep("LaguerrePolynomial")),
        # CreateStep.LegendrePolynomial(context, order=2),
        # Debugger.log(GetStep("LegendrePolynomial")),
        # CreateStep.Inverse(context),
        # Debugger.log(GetStep("Inverse")),
        # CreateStep.Orthonormalize(context),
        # Debugger.log(GetStep("Orthonormalize")),
        # CreateStep.Autocorrelate(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]]),
        # Debugger.log(GetStep("Autocorrelate")),
        # CreateStep.Covariance(context, direction=0),
        # Debugger.log(GetStep("Covariance")),
        # CreateStep.CorrelationCoefficient(context, direction=1),
        # Debugger.log(GetStep("CorrelationCoefficient")),
        # CreateStep.RollingQuotient(context[:, ColumnIndex.FIRST], period=3),
        # Debugger.log(GetStep("RollingQuotient")),
        # CreateStep.RollingProduct(context[:, ColumnIndex.FIRST], period=3),
        # Debugger.log(GetStep("RollingProduct")),
        # CreateStep.RollingSum(context[:, ColumnIndex.FIRST], period=3),
        # Debugger.log(GetStep("RollingSum")),
        # CreateStep.RollingDifference(context[:, ColumnIndex.FIRST], period=3),
        # Debugger.log(GetStep("RollingDifference"))
    ])


if __name__ == "__main__":
    example_flow()
