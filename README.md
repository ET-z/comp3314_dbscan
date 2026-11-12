# comp3314_dbscan - DBSCAN Reproduction

A from-scratch implementation and experimental study of DBSCAN for the HKU Introduction to Machine Learning course project.

## Paper

Ester, M., Kriegel, H.-P., Sander, J., & Xu, X. (1996).

**A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise.**

Proceedings of KDD-96, pp. 226–231.

## Objectives

- Understand the DBSCAN algorithm.
- Implement DBSCAN from scratch using Python.
- Recreate small examples inspired by the paper.
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

Runs DBSCAN, K-means and CLARANS on three small datasets:

- `compact`: four round clusters with different sizes (104 points).
- `nonconvex`: four curved clusters (160 points).
- `noisy`: mixed shapes with 15 added outliers (147 points).

These are our own examples inspired by Figures 1, 5 and 6 of the paper,
not the original coordinates. `datasets.py` generates them with seed 42,
and `main.py` saves copies in `data/` with columns `x,y,label` (`-1` is noise).
The labels are only used for evaluation, not given to the algorithms.

DBSCAN uses MinPts=4 and eps=0.55, 0.22 and 0.32 respectively. Both
baselines get the correct cluster count, k=4. These settings were chosen
for the examples, so they don't show that DBSCAN always works this well.

The script prints cluster counts, noise counts, ARI and median runtime
from three runs after warm-up, using one numerical thread. ARI includes
noise as a label. These small datasets aren't a runtime scaling benchmark.

Charts and `results.csv` go in `figures/synthetic/`. Each comparison shows
the generated labels beside the three algorithms. To just save the charts:

```bash
python main.py --no-show
```

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

See `figures/synthetic/results.csv` for the latest scores and timings.
The comparison charts show where the algorithms agree with the generated labels.
