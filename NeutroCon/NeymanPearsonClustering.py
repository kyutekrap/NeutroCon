import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def NeymanPearsonClustering(context: np.ndarray, n_clusters: int, threshold: float) -> tuple:
    """
    Applies a Neyman-Pearson clustering approach.

    Parameters:
    - context (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
    - n_clusters (int): The number of clusters to form.
    - threshold (float): The threshold for cluster assignment in the Neyman-Pearson strategy.

    Returns:
    - projected_matrix (numpy.ndarray): The data matrix where each point is replaced by its cluster centroid.
    - labels (numpy.ndarray): Cluster labels for each data point, of shape (n_samples,).
    - centroids (numpy.ndarray): The final centroids found by the algorithm, of shape (n_clusters, n_features).
    """
    n_samples, n_features = context.shape

    np.random.seed(42)

    probabilities = np.random.rand(n_samples, n_clusters)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    labels = np.argmax(probabilities > threshold, axis=1)

    new_centroids = np.array([context[labels == i].mean(axis=0) for i in range(n_clusters)])

    projected_matrix = np.zeros_like(context)
    for i in range(n_clusters):
        projected_matrix[labels == i] = new_centroids[i]

    return projected_matrix, labels, new_centroids
