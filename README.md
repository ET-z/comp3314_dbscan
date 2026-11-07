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

Generates the two-moons dataset, runs DBSCAN with `eps=0.2` and
`min_pts=4`, prints cluster/noise counts, and displays the original data
and clustering result. Figures are saved to `figures/two_moons.png` and
`figures/two_moons_dbscan.png`.

The supplied implementation also includes its own demo with a k-distance
graph and suggested epsilon:

```bash
python dbscan.py
```

## Run the tests

```bash
python test.py
```

The supplied runner prints a result for each test and exits with a nonzero
status if any test fails. It covers basic clustering, comparison with
scikit-learn, point-order behavior, cluster definitions, and parameter behavior.

Run the supplied performance benchmark separately:

```bash
python test.py --bench
```

## Project Status

The groupmate's implementation is in `dbscan.py` and the supplied tests
are in `test.py`. Both files are preserved unchanged.

Current known failure: with SciPy installed, all 11 supplied tests fail with
`TypeError: __init__() got an unexpected keyword argument 'leaf_size'`
from the `cKDTree` call in `_region_queries`. Both demos and the benchmark
also use this code path, so they currently stop with the same error before
producing clustering results. The original-data figure from `main.py` is
saved before that call. This bug is intentionally left for later correction.

## Results
