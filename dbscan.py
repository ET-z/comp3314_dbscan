"""
DBSCAN - Density-Based Spatial Clustering of Applications with Noise
Reference: Ester, Kriegel, Sander, Xu.
    "A Density-Based Algorithm for Discovering Clusters in Large
     Spatial Databases with Noise", KDD-96.

Label convention:
    UNCLASSIFIED = -2   not yet classified
    NOISE        = -1   noise (Definition 6)
    >= 0                cluster id (Definition 5)
"""

import numpy as np

UNCLASSIFIED = -2
NOISE = -1


# ----------------------------------------------------------------------
# 1. Region query (regionQuery)
# ----------------------------------------------------------------------
def _region_queries(X, eps, leaf_size=30):
    """
    Return the Eps-neighborhood of every point (including the point
    itself, since dist(p,p)=0 <= Eps), i.e. N_Eps(p) from Definition 1.
    Uses a KD-tree when scipy is available; otherwise falls back to
    a brute-force implementation.
    """
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(X, leaf_size=leaf_size)
        return [tree.query_ball_point(X[i], r=eps) for i in range(len(X))]
    except ImportError:
        return _region_queries_bf(X, eps)


def _region_queries_bf(X, eps, block=512):
    """Brute-force O(n^2) version for environments without scipy."""
    n = len(X)
    out = []
    for s in range(0, n, block):
        e = min(s + block, n)
        diff = X[s:e, None, :] - X[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=-1))
        for row in d:
            out.append(np.flatnonzero(row <= eps).tolist())
    return out


# ----------------------------------------------------------------------
# 2. ExpandCluster - pseudocode on page 6 of the paper
# ----------------------------------------------------------------------
def _expand_cluster(labels, neighborhoods, p, cluster_id, min_pts):
    """
    Starting from core point p, retrieve all points density-reachable
    from p (Lemma 2). Seeds are stored as a list plus a dedup set so
    that the result is independent of the point visit order.
    """
    labels[p] = cluster_id
    seeds = [p]
    in_seeds = {p}
    head = 0

    while head < len(seeds):
        q = seeds[head]
        head += 1

        # q is not a core point -> absorb it as a border point only.
        if len(neighborhoods[q]) < min_pts:
            continue

        for r in neighborhoods[q]:
            if labels[r] == UNCLASSIFIED:
                labels[r] = cluster_id
                if r not in in_seeds:
                    seeds.append(r)
                    in_seeds.add(r)
            elif labels[r] == NOISE:
                # A point previously marked as NOISE may in fact be a
                # border point of this cluster (see page 7 of the paper).
                labels[r] = cluster_id


# ----------------------------------------------------------------------
# 3. DBSCAN main loop - pseudocode on page 6 of the paper
# ----------------------------------------------------------------------
def dbscan(X, eps, min_pts, labels=None):
    """
    Parameters
    ----------
    X       : (n, d) sample matrix
    eps     : radius of the Eps-neighborhood
    min_pts : core point condition |N_Eps(q)| >= MinPts
              (4 is recommended for 2-D data)

    Returns
    -------
    labels : (n,) array, -1 for noise, 0..k-1 for cluster ids
    """
    X = np.ascontiguousarray(X, dtype=float)
    n = X.shape[0]
    labels = (np.full(n, UNCLASSIFIED, dtype=int)
              if labels is None else np.asarray(labels, dtype=int).copy())

    neighborhoods = _region_queries(X, eps)

    cluster_id = 0
    for i in range(n):
        if labels[i] != UNCLASSIFIED:
            continue
        if len(neighborhoods[i]) < min_pts:
            labels[i] = NOISE          # not a core point -> tentative noise
            continue
        _expand_cluster(labels, neighborhoods, i, cluster_id, min_pts)
        cluster_id += 1

    return labels


# ----------------------------------------------------------------------
# 4. Parameter selection: k-dist graph (Section 4.2 of the paper)
# ----------------------------------------------------------------------
def k_dist_values(X, k):
    """
    Return the k-dist values in descending order.
    k-dist(p) = distance from p to its k-th nearest neighbor,
                excluding p itself.
    """
    X = np.asarray(X, dtype=float)
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(X).query(X, k=k + 1)      # column 0 is the point itself (0)
        kd = d[:, k]
    except ImportError:
        diff = X[:, None, :] - X[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        d.sort(axis=1)
        kd = d[:, k]
    return np.sort(kd)[::-1]


def suggest_eps_by_noise_ratio(X, noise_ratio, k=4):
    """
    If the user can estimate the noise ratio, take the k-dist value at
    position (noise_ratio * n) as Eps. This corresponds to step 2 of
    the interactive procedure described in Section 4.2.
    """
    kd = k_dist_values(X, k)
    idx = int(np.clip(round(noise_ratio * len(kd)), 0, len(kd) - 1))
    return float(kd[idx])


# ----------------------------------------------------------------------
# 5. Demo
# ----------------------------------------------------------------------
def _make_demo(seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal([2, 2], 0.45, (180, 2))                # compact blob
    b = rng.normal([8, 2], 0.85, (180, 2))                # sparser blob
    t = np.linspace(0.2, np.pi - 0.2, 150)                # non-convex half-moon
    c = np.c_[5 + 3 * np.cos(t), 9 + 3 * np.sin(t)] + rng.normal(0, 0.12, (150, 2))
    d = rng.uniform([-2, -2], [14, 14], (40, 2))          # noise
    return np.vstack([a, b, c, d])


if __name__ == "__main__":
    X = _make_demo()

    eps = suggest_eps_by_noise_ratio(X, noise_ratio=40 / len(X), k=4)
    print(f"Suggested Eps = {eps:.3f}")

    labels = dbscan(X, eps=eps, min_pts=4)

    n_clusters = labels.max() + 1
    n_noise = int((labels == NOISE).sum())
    print(f"Clusters found: {n_clusters}, noise points: {n_noise}")
    for cid in range(n_clusters):
        print(f"  cluster {cid}: {(labels == cid).sum()} points")

    # Visualization
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5))

        kd = k_dist_values(X, k=4)
        ax[0].plot(kd, lw=1.2)
        ax[0].axhline(eps, color="r", ls="--", label=f"Eps={eps:.2f}")
        ax[0].set_title("sorted 4-dist graph")
        ax[0].set_xlabel("points (sorted)")
        ax[0].set_ylabel("4-dist")
        ax[0].legend()

        colors = np.array(["#cccccc", "#1f77b4", "#ff7f0e", "#2ca02c",
                           "#d62728", "#9467bd", "#8c564b"])
        ax[1].scatter(X[:, 0], X[:, 1], s=14,
                      c=colors[np.where(labels < 0, 0, labels % 6 + 1)],
                      edgecolors="none")
        ax[1].set_title(f"DBSCAN result (eps={eps:.2f}, MinPts=4)")
        ax[1].set_aspect("equal")
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass