import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def KMeansClustering(context: np.ndarray, n_clusters: int, max_iters: int = 100, tol: float = 1e-4) -> tuple:
    """
    Applies K-means clustering.

    Parameters:
    - context (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
    - n_clusters (int): The number of clusters to form.
    - max_iters (int): The maximum number of iterations to run the K-means algorithm.
    - tol (float): The tolerance to declare convergence.

    Returns:
    - projected_matrix (numpy.ndarray): The data matrix where each point is replaced by its cluster centroid.
    - labels (numpy.ndarray): Cluster labels for each data point, of shape (n_samples,).
    - centroids (numpy.ndarray): The final centroids found by the algorithm, of shape (n_clusters, n_features).
    """
    n_samples, n_features = context.shape

    np.random.seed(42)
    initial_indices = np.random.choice(n_samples, n_clusters, replace=False)
    centroids = context[initial_indices]

    for iteration in range(max_iters):
        distances = np.linalg.norm(context[:, np.newaxis] - centroids, axis=2)

        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([context[labels == i].mean(axis=0) for i in range(n_clusters)])

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    projected_matrix = np.zeros_like(context)
    for i in range(n_clusters):
        projected_matrix[labels == i] = centroids[i]

    return projected_matrix, labels, centroids
