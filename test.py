"""
DBSCAN test suite.
Usage:
    python test_dbscan.py            # run all tests
    python test_dbscan.py --bench    # run performance benchmark
    pytest test_dbscan.py -v         # run via pytest
"""
import sys
import time
from collections import deque

import numpy as np

from dbscan import (dbscan, k_dist_values, suggest_eps_by_noise_ratio,
                    NOISE, _region_queries)

RNG = np.random.default_rng(42)


# ====================== Helper functions ======================
def _grid(cx, cy, step=0.5):
    """3x3 grid; maximum pairwise distance is sqrt(2) ~= 1.414."""
    return np.array([[cx + i * step, cy + j * step]
                     for i in (-1, 0, 1) for j in (-1, 0, 1)])


def _blobs(n=300, centers=3, std=0.4, seed=0, noise=20):
    rng = np.random.default_rng(seed)
    cs = rng.uniform(-8, 8, (centers, 2))
    X = np.vstack([rng.normal(c, std, (n // centers, 2)) for c in cs])
    if noise:
        X = np.vstack([X, rng.uniform(X.min(0) - 1, X.max(0) + 1, (noise, 2))])
    return X


def _core_mask(X, eps, mp):
    return np.array([len(n) >= mp for n in _region_queries(X, eps)])


# ====================== 1. Basic unit tests ======================
def test_two_clusters_and_noise():
    """Two separated clusters + 1 outlier: expect 2 clusters of 9 + 1 noise."""
    X = np.vstack([_grid(0, 0), _grid(10, 0), [[5.0, 5.0]]])
    lab = dbscan(X, eps=1.5, min_pts=3)
    assert lab.max() + 1 == 2, lab
    assert (lab == NOISE).sum() == 1
    assert (lab == 0).sum() == 9 and (lab == 1).sum() == 9
    assert lab[-1] == NOISE, "isolated point must be labeled as noise (Def. 6)"


def test_all_noise():
    """All pairwise distances > eps: every point must be noise."""
    X = np.array([[0., 0.], [10., 10.], [20., 0.]])
    assert np.all(dbscan(X, eps=0.5, min_pts=3) == NOISE)


def test_single_point():
    """Single point: core point if MinPts=1 (it is in its own
    neighborhood), noise if MinPts=2."""
    X = np.array([[1.0, 2.0]])
    assert dbscan(X, eps=0.5, min_pts=1)[0] == 0
    assert dbscan(X, eps=0.5, min_pts=2)[0] == NOISE


def test_duplicate_points():
    """10 coincident points: all pairwise distances are 0 <= eps,
    so they must form a single cluster (border point handling)."""
    X = np.zeros((10, 2))
    assert np.all(dbscan(X, eps=0.5, min_pts=3) == 0)


def test_empty_input():
    """Empty database must return an empty label array without error."""
    lab = dbscan(np.empty((0, 2)), eps=1.0, min_pts=3)
    assert lab.shape == (0,)


def test_large_eps_single_cluster():
    """With a huge eps, all points merge into one cluster."""
    X = _blobs(n=100, centers=3, std=0.3, seed=0, noise=0)
    assert np.all(dbscan(X, eps=100.0, min_pts=5) == 0)


# ====================== 2. Cross-check against sklearn ======================
def test_against_sklearn():
    """
    The core point sets must match exactly; on core points the cluster
    assignment must have ARI ~= 1. (Border point assignment can be
    ambiguous, so we only compare core points.)
    """
    try:
        from sklearn.cluster import DBSCAN as SkDBSCAN
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        print("      [skip] scikit-learn is not installed")
        return

    for seed in range(5):
        X = _blobs(n=400, centers=4, std=0.6, seed=seed, noise=30)
        eps, mp = 0.8, 5
        ours = dbscan(X, eps, mp)
        sk = SkDBSCAN(eps=eps, min_samples=mp).fit(X)

        sk_core = np.zeros(len(X), bool)
        sk_core[sk.core_sample_indices_] = True
        our_core = _core_mask(X, eps, mp)

        assert np.array_equal(our_core, sk_core), \
            f"seed={seed}: core point sets differ"
        ari = adjusted_rand_score(ours[our_core], sk.labels_[our_core])
        assert ari > 0.999, f"seed={seed}: core point partition ARI={ari:.4f}"


# ====================== 3. Independence of point visit order ======================
def test_permutation_invariance():
    """Shuffle the points; the noise set and cluster size distribution
    must stay the same (page 7 of the paper)."""
    X = _blobs(n=300, centers=3, std=0.3, seed=7, noise=15)
    base = dbscan(X, eps=0.7, min_pts=5)
    sizes_base = sorted(np.bincount(base[base >= 0]).tolist())

    for _ in range(5):
        perm = RNG.permutation(len(X))
        lab = dbscan(X[perm], eps=0.7, min_pts=5)
        lab_back = np.empty_like(lab)
        lab_back[perm] = lab          # map back to original indices

        assert np.array_equal(base == NOISE, lab_back == NOISE)
        sizes = sorted(np.bincount(lab_back[lab_back >= 0]).tolist())
        assert sizes == sizes_base, f"cluster sizes changed: {sizes} vs {sizes_base}"


# ====================== 4. Per-cluster verification of Def. 5 / Def. 6 ======================
def verify_definitions(X, labels, eps, mp):
    """
    For every discovered cluster, verify:
      (1) Maximality: every point density-reachable from a core point
          of the cluster also belongs to the cluster.
      (2) Connectivity: all core points of the cluster lie in a single
          connected component of the "core point neighborhood graph".
    Border points are covered automatically by the union of core
    neighborhoods.
    """
    nb = _region_queries(X, eps)
    core = _core_mask(X, eps, mp)

    for cid in range(labels.max() + 1):
        members = set(np.flatnonzero(labels == cid).tolist())
        cores = [i for i in members if core[i]]
        assert cores, f"cluster {cid} has no core point; violates Definition 5"

        # BFS over core points
        seen, dq = {cores[0]}, deque([cores[0]])
        while dq:
            u = dq.popleft()
            for v in nb[u]:
                if core[v] and v not in seen:
                    seen.add(v)
                    dq.append(v)
        assert seen.issuperset(cores), \
            f"cluster {cid} core points are not connected (Def. 5-2)"

        # Maximality: union of these core neighborhoods == cluster members
        reach = set()
        for u in seen:
            reach.update(nb[u])
        assert reach == members, \
            f"cluster {cid} violates maximality (Def. 5-1)"


def test_cluster_definitions():
    """Verify both conditions of Definition 5 on random data."""
    for seed in range(5):
        X = _blobs(n=300, centers=3, std=0.35, seed=seed, noise=10)
        eps, mp = 0.6, 4
        lab = dbscan(X, eps, mp)
        verify_definitions(X, lab, eps, mp)


# ====================== 5. Parameter behavior ======================
def test_eps_monotonic_noise():
    """As Eps grows, the number of noise points must not increase
    (the sparsest-cluster density parameter is a lower bound)."""
    X = _blobs(n=400, centers=4, std=0.5, seed=3, noise=40)
    prev = None
    for eps in (0.3, 0.5, 0.8, 1.2, 2.0):
        n_noise = int((dbscan(X, eps, 5) == NOISE).sum())
        if prev is not None:
            assert n_noise <= prev, \
                f"eps={eps}: noise increased from {prev} to {n_noise}"
        prev = n_noise


def test_kdist_and_suggest_eps():
    """k-dist values must be sorted in descending order, and the
    suggested Eps should yield at least 2 clusters."""
    X = np.vstack([_grid(0, 0, 0.3), _grid(5, 0, 0.3),
                   RNG.uniform(-3, 8, (30, 2))])
    kd = k_dist_values(X, 4)
    assert np.all(np.diff(kd) <= 1e-12), "k-dist must be descending"
    eps = suggest_eps_by_noise_ratio(X, noise_ratio=30 / len(X), k=4)
    assert eps > 0
    lab = dbscan(X, eps, 4)
    assert lab.max() + 1 >= 2, \
        f"suggested eps={eps:.3f} failed to separate clusters"


# ====================== 6. Performance benchmark ======================
def benchmark():
    """Check near O(n log n) scaling: doubling n should multiply the
    runtime by roughly 2.2-2.5x."""
    print(f"{'n':>8} {'time(s)':>10} {'t/(n*log n)':>14}")
    for n in (500, 1000, 2000, 4000, 8000):
        X = _blobs(n=n, centers=8, std=0.5, seed=0, noise=n // 10)
        t0 = time.perf_counter()
        dbscan(X, eps=0.8, min_pts=4)
        dt = time.perf_counter() - t0
        print(f"{n:>8} {dt:>10.3f} {dt / (n * np.log(n)):>14.3e}")


# ====================== Runner ======================
def run_all():
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    if "--bench" in sys.argv:
        benchmark()
    else:
        sys.exit(0 if run_all() else 1)
