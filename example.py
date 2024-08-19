from Link import *
import logging
import numpy as np
from enum import IntEnum
import NeutroCon

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Create a matrix context
context = np.random.rand(3, 3)


class ColumnIndex(IntEnum):
    FIRST = 0
    SECOND = 1
    THIRD = 2


@register_flow(Debug=False)
def example_flow():
    CreateFlow([
        # CreateStep.Add(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Add")),
        # CreateStep.Divide(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Divide")),
        # CreateStep.Multiply(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Multiply")),
        # CreateStep.Subtract(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=0),
        # Debugger.log(GetStep("Subtract")),
        # CreateStep.Average(context[:, [ColumnIndex.FIRST, ColumnIndex.SECOND]], direction=1),
        # Debugger.log(GetStep("Average")),
        # CreateStep.Normalize(context[:, [ColumnIndex.FIRST]]),
        # Debugger.log(GetStep("Normalize")),
        # CreateStep.WeightedAverage(context),
        # Debugger.log(GetStep("WeightedAverage")),
        # CreateStep.HermitePolynomial(context[:, [ColumnIndex.FIRST]], order=2),
        # Debugger.log(GetStep("HermitePolynomial")),
        # CreateStep.LaguerrePolynomial(context, order=2),
        # Debugger.log(GetStep("LaguerrePolynomial")),
        # CreateStep.LegendrePolynomial(context, order=2),
        # Debugger.log(GetStep("LegendrePolynomial")),
        # CreateStep.Inverse(context),
        # Debugger.log(GetStep("Inverse")),
        # CreateStep.Orthonormalize(context),
        # Debugger.log(GetStep("Orthonormalize")),
        # CreateStep.Autocorrelate(context[:, [ColumnIndex.FIRST]]),
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
        # CreateStep.RollingDifference(context[:, ColumnIndex.FIRST], period=2),
        # Debugger.log(GetStep("RollingDifference")),
        # CreateStep.Integral(context, direction=0),
        # Debugger.log(GetStep("Integral")),
        # CreateStep.EigenvalueEigenvector(context),
        # Debugger.log(GetStep("EigenvalueEigenvector")),
        # CreateStep.KLEigenvector(context),
        # Debugger.log(GetStep("KLEigenvector")),
        # CreateStep.LeastSquare(context),
        # Debugger.log(GetStep("LeastSquare")),
        # CreateStep.GaussElimination(context),
        # Debugger.log(GetStep("GaussElimination")),
        # CreateStep.LowPassFilter(context),
        # Debugger.log(GetStep("LowPassFilter")),
        # CreateStep.MinmaxNormalize(context, direction=0),
        # Debugger.log(GetStep("MinmaxNormalize")),
        # CreateStep.MinimaxClustering(context, 3),
        # Debugger.log(GetStep("MinimaxClustering")),
        # CreateStep.KMeansClustering(context, 3),
        # Debugger.log(GetStep("KMeansClustering")),
        # CreateStep.NeymanPearsonClustering(context, 3, 0.5),
        # Debugger.log(GetStep("NeymanPearsonClustering"))
    ])


if __name__ == "__main__":
    example_flow()
