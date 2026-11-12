"""K-means baseline using scikit-learn."""

from sklearn.cluster import KMeans


def kmeans(X, n_clusters, random_state=42, n_init=10):
    """
    Parameters
    ----------
    X            : (n, d) sample matrix
    n_clusters   : number of clusters to find
    random_state : seed for reproducible initialization
    n_init       : number of initializations; keep the best result

    Returns
    -------
    labels : (n,) array, 0..k-1 for cluster ids (no noise label)
    """
    model = KMeans(n_clusters=n_clusters, n_init=n_init,
                   random_state=random_state)
    return model.fit_predict(X)
