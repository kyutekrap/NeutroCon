import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def Partition(context: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Partition a matrix according to the provided labels.

    Parameters:
    - context (np.ndarray): The input matrix where each row represents a data point.
    - labels (np.ndarray): An array of labels where each entry corresponds to the label for the row in the matrix.

    Returns:
    - partitions (np.ndarray): Matrices of rows from the original matrix corresponding to each label.
    """
    if len(context) != len(labels):
        raise ValueError("The length of labels must match the number of rows in the matrix.")

    unique_labels = np.unique(labels)
    partitions = []

    for label in unique_labels:
        partitions.append(context[labels == label])

    return np.array(partitions)
