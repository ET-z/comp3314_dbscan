"""
CLARANS - Clustering Large Applications based upon RANdomized Search.
Simple k-medoids baseline with random swaps and multiple restarts.
Reference: Ng and Han, "CLARANS: A Method for Clustering Objects
    for Spatial Data Mining", IEEE TKDE, 2002.
"""

from numbers import Integral

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.utils.validation import check_array


def clarans(X, n_clusters, num_local=2, max_neighbors=50, random_state=42):
    """
    Parameters
    ----------
    X             : (n, d) sample matrix
    n_clusters    : number of medoids (representative data points)
    num_local     : number of random restarts
    max_neighbors : stop a search after this many consecutive rejected swaps
    random_state  : seed for reproducible initialization and swaps

    Returns
    -------
    labels : (n,) array, 0..k-1 for cluster ids (no noise label)

    Minimize the sum of Euclidean distances to the nearest medoid.
    Distances are cached in an (n, n) matrix, so this simple version is
    intended for small comparison datasets rather than large databases.
    """
    X = check_array(X, dtype=float)
    n = len(X)
    for name, value in (("n_clusters", n_clusters), ("num_local", num_local),
                        ("max_neighbors", max_neighbors)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if n_clusters > n:
        raise ValueError("n_clusters must not exceed the number of samples")
    if n_clusters == n:
        return np.arange(n)

    rng = np.random.default_rng(random_state)
    distances = pairwise_distances(X, metric="euclidean")
    best_cost = np.inf
    best_medoids = None

    for _ in range(num_local):
        medoids = rng.choice(n, size=n_clusters, replace=False)
        non_medoids = np.setdiff1d(np.arange(n), medoids)
        cost = distances[:, medoids].min(axis=1).sum()
        rejected = 0

        while rejected < max_neighbors:
            # A neighbor replaces one medoid with one non-medoid.
            medoid_pos = rng.integers(n_clusters)
            candidate_pos = rng.integers(len(non_medoids))
            neighbor = medoids.copy()
            neighbor[medoid_pos] = non_medoids[candidate_pos]
            neighbor_cost = distances[:, neighbor].min(axis=1).sum()

            if neighbor_cost < cost:
                non_medoids[candidate_pos] = medoids[medoid_pos]
                medoids = neighbor
                cost = neighbor_cost
                rejected = 0
            else:
                rejected += 1

        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids.copy()

    labels = distances[:, best_medoids].argmin(axis=1)
    # Keep each medoid in its own cluster when duplicate points tie.
    labels[best_medoids] = np.arange(n_clusters)
    return labels
