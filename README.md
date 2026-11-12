# comp3314_dbscan - DBSCAN Reproduction

A from-scratch implementation and experimental study of DBSCAN for the HKU Introduction to Machine Learning course project.

## Paper

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).

**A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.**

Proceedings of KDD-96, pp. 226–231.

## Objectives

- Understand the DBSCAN algorithm.
- Implement DBSCAN from scratch using Python.
- Reproduce selected experiments from the paper.
- Compare DBSCAN with other clustering algorithms.
- Investigate limitations and possible improvements.

## Setup

Create and activate a virtual environment (or activate the existing `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the demo

```bash
python main.py
```

Generates the same two-moons dataset for DBSCAN, K-means, and CLARANS.
DBSCAN uses `eps=0.2` and `min_pts=4`; the other algorithms are given
`n_clusters=2`. Random seeds are fixed at 42 for reproducibility.

The demo prints cluster/noise counts, adjusted Rand index (ARI) against
the generated labels, and clustering runtime in seconds. ARI includes all
points, including any DBSCAN noise points; 1.0 means a perfect match.
Timings are single runs on this small dataset, not a scaling benchmark.

Figures are saved in `figures/`: the original `two_moons.png`, individual
`two_moons_dbscan.png`, `two_moons_kmeans.png`, and `two_moons_clarans.png`
plots, plus a side-by-side `two_moons_comparison.png`.

## Baseline algorithms

Both baselines follow the `dbscan.py` interface: pass a sample matrix and
receive a NumPy array of integer cluster labels.

```python
from kmeans import kmeans
from clarans import clarans

kmeans_labels = kmeans(X, n_clusters=2, random_state=42)
clarans_labels = clarans(X, n_clusters=2, num_local=2,
                        max_neighbors=50, random_state=42)
```

- `kmeans.py` wraps [scikit-learn's KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  with 10 initializations by default.
- `clarans.py` implements the random medoid swaps and restarts from
  [Ng and Han's CLARANS algorithm](https://cs.ecu.edu/dingq/CSCI6905/readings/CLARANS.pdf)
  using NumPy and scikit-learn's Euclidean distances. `num_local` controls
  restarts; `max_neighbors` limits consecutive rejected swaps per search.
  It caches an O(n²) distance matrix, so it is intended for small experiments.

Both require a cluster count and assign every point to a cluster. All
dependencies are already included in `requirements.txt`.

DBSCAN also includes its own demo with a k-distance
graph and suggested epsilon:

```bash
python dbscan.py
```

## Run the tests

```bash
python test.py
python test_clustering.py
```

The supplied runner prints a result for each test and exits with a nonzero
status if any test fails. It covers basic clustering, comparison with
scikit-learn, point-order behavior, cluster definitions, and parameter behavior.
The baseline tests cover separated clusters, reproducibility, duplicate points,
cluster-count limits, and invalid CLARANS parameters.

Run the supplied performance benchmark separately:

```bash
python test.py --bench
```

## Project Status

The DBSCAN implementation is in `dbscan.py`, with baselines in `kmeans.py`
and `clarans.py`. The original DBSCAN implementation is unchanged by the
baseline comparison work. The supplied scikit-learn cross-check in `test.py`
uses the fitted estimator to access core sample indices and labels.

## Results

On the default two-moons dataset with the parameters above:

| Algorithm | Clusters | Noise points | ARI |
| --- | ---: | ---: | ---: |
| DBSCAN | 2 | 0 | 1.000 |
| K-means | 2 | 0 | 0.254 |
| CLARANS | 2 | 0 | 0.311 |

These results compare recovery of the two curved moons on this dataset;
they do not establish performance across other cluster shapes or datasets.
