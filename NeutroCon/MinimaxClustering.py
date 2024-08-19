import numpy as np
from Link import register_step
from ._config import Debug


@register_step(Debug=Debug)
def MinimaxClustering(context: np.ndarray, n_clusters: int, max_iters: int = 100, tol: float = 1e-4) -> tuple:
    """
    Applies minimax clustering.

    Parameters:
    - context (numpy.ndarray): The input data matrix of shape (n_samples, n_features).
    - n_clusters (int): The number of clusters to form.
    - max_iters (int): The maximum number of iterations to run the minimax clustering algorithm.
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
        labels = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            distances = np.zeros(n_clusters)
            for c in range(n_clusters):
                cluster_points = context[labels == c]
                if cluster_points.shape[0] > 0:
                    distances[c] = np.max(np.linalg.norm(context[i] - cluster_points, axis=1))
                else:
                    distances[c] = np.inf
            labels[i] = np.argmin(distances)

        new_centroids = np.zeros_like(centroids)
        for c in range(n_clusters):
            cluster_points = context[labels == c]
            if len(cluster_points) > 0:
                max_distances = np.array([np.max(np.linalg.norm(cluster_points - p, axis=1)) for p in cluster_points])
                new_centroids[c] = cluster_points[np.argmin(max_distances)]

        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    projected_matrix = np.zeros_like(context)
    for i in range(n_clusters):
        projected_matrix[labels == i] = centroids[i]

    return projected_matrix, labels, centroids
