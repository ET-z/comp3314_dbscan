import argparse
import csv
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from clarans import clarans
from datasets import make_datasets
from dbscan import NOISE, dbscan
from kmeans import kmeans


ROOT = Path(__file__).resolve().parent
FIGURES_DIR = ROOT / "figures" / "synthetic"


def plot_clusters(ax, X, labels, title):
    clustered = labels != NOISE
    # Use a separate colour for each cluster, and grey crosses for noise.
    colours = plt.get_cmap("tab10")(labels[clustered] % 10)
    ax.scatter(X[clustered, 0], X[clustered, 1], c=colours, s=18)
    if (~clustered).any():
        ax.scatter(X[~clustered, 0], X[~clustered, 1], c="gray",
                   marker="x", s=22, label="Noise")
        ax.legend(fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_aspect("equal")


def main():
    parser = argparse.ArgumentParser(description="Compare clustering on three synthetic datasets")
    parser.add_argument("--no-show", action="store_true", help="save charts without opening windows")
    args = parser.parse_args()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)
    datasets = make_datasets()
    results = []

    # Warm up the imports and keep numerical thread counts the same.
    with threadpool_limits(limits=1):
        warmup = datasets[0][1][:30]
        dbscan(warmup, eps=0.55, min_pts=4)
        kmeans(warmup, n_clusters=4)
        clarans(warmup, n_clusters=4)

        for name, X, y, eps in datasets:
            np.savetxt(ROOT / "data" / f"{name}.csv", np.column_stack((X, y)),
                       delimiter=",", header="x,y,label", comments="", fmt=["%.8f", "%.8f", "%d"])
            print(f"\n{name}: {len(X)} points, eps={eps}, MinPts=4, baseline k=4")
            print(f"{'Algorithm':<10} {'Clusters':>8} {'Noise':>7} {'ARI':>8} {'Time (s)':>10}")
            fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharex=True, sharey=True)
            plot_clusters(axes[0], X, y, "Generated labels")
            algorithms = [
                ("DBSCAN", lambda: dbscan(X, eps=eps, min_pts=4)),
                ("K-means", lambda: kmeans(X, n_clusters=4, random_state=42)),
                ("CLARANS", lambda: clarans(X, n_clusters=4, random_state=42)),
            ]
            for ax, (algorithm, run) in zip(axes[1:], algorithms):
                times = []
                for _ in range(3):
                    start = perf_counter()
                    labels = run()
                    times.append(perf_counter() - start)
                elapsed = float(np.median(times))
                clusters = len(set(labels) - {NOISE})
                noise = int(np.sum(labels == NOISE))
                ari = adjusted_rand_score(y, labels)
                print(f"{algorithm:<10} {clusters:>8} {noise:>7} {ari:>8.3f} {elapsed:>10.6f}")
                results.append([name, algorithm, len(X), eps, 4, 4,
                                clusters, noise, ari, elapsed])
                plot_clusters(ax, X, labels, f"{algorithm}\nARI = {ari:.3f}, noise = {noise}")
            fig.suptitle(f"{name.capitalize()} — our synthetic example (eps={eps}, MinPts=4, k=4)")
            fig.text(0.5, 0.02, "Inspired by Ester et al. (1996); not the original paper's datasets",
                     ha="center", fontsize=9)
            fig.subplots_adjust(left=0.05, right=0.99, bottom=0.16, top=0.76, wspace=0.3)
            fig.savefig(FIGURES_DIR / f"{name}_comparison.png", dpi=180)

    with (FIGURES_DIR / "results.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "algorithm", "n_points", "eps", "min_pts", "baseline_k",
                         "clusters", "noise_points", "ari", "median_seconds"])
        writer.writerows(results)

    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(len(datasets))
    for i, algorithm in enumerate(["DBSCAN", "K-means", "CLARANS"]):
        times = [r[-1] for r in results if r[1] == algorithm]
        ax.bar(positions + (i - 1) * 0.25, np.array(times) * 1000,
               width=0.25, label=algorithm)
    ax.set_xticks(positions, [item[0] for item in datasets])
    ax.set_ylabel("Clustering time (milliseconds)")
    ax.set_title("Our synthetic examples: median of 3 runs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "runtime.png", dpi=180)
    print(f"\nCharts and results saved to {FIGURES_DIR}")
    print("These small examples demonstrate behaviour, not runtime scaling.")
    if not args.no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    main()
