
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

from clarans import clarans
from dbscan import NOISE, dbscan
from kmeans import kmeans


FIGURES_DIR = Path(__file__).parent / "figures"


def _plot_clusters(ax, X, labels, title):
    clustered = labels != NOISE
    ax.scatter(X[clustered, 0], X[clustered, 1],
               c=labels[clustered], cmap="tab10", s=15)
    if (~clustered).any():
        ax.scatter(X[~clustered, 0], X[~clustered, 1],
                   c="gray", marker="x", s=15, label="Noise")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    # Generate a simple 2D clustering dataset
    X, y = make_moons(
        n_samples=300,
        noise=0.07,
        random_state=42
    )
    
    # Visualize the dataset
    plt.figure(figsize=(6, 4))
    plt.scatter(X[:, 0], X[:, 1], s=15)
    plt.title("Two Moons — Original Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()

    plt.savefig(FIGURES_DIR / "two_moons.png", dpi=150)

    eps, min_pts, k = 0.2, 4, 2
    algorithms = [
        ("DBSCAN", f"eps={eps}, MinPts={min_pts}",
         lambda: dbscan(X, eps=eps, min_pts=min_pts)),
        ("K-means", f"k={k}",
         lambda: kmeans(X, n_clusters=k, random_state=42)),
        ("CLARANS", f"k={k}, restarts=2, neighbors=50",
         lambda: clarans(X, n_clusters=k, num_local=2,
                         max_neighbors=50, random_state=42)),
    ]

    comparison, axes = plt.subplots(1, 3, figsize=(16, 4),
                                    sharex=True, sharey=True)
    print(f"{'Algorithm':<10} {'Clusters':>8} {'Noise':>7} {'ARI':>8} {'Time (s)':>10}")
    for ax, (name, params, run) in zip(axes, algorithms):
        start = perf_counter()
        labels = run()
        elapsed = perf_counter() - start
        n_clusters = len(set(labels) - {NOISE})
        n_noise = int((labels == NOISE).sum())
        ari = adjusted_rand_score(y, labels)
        print(f"{name:<10} {n_clusters:>8} {n_noise:>7} {ari:>8.3f} {elapsed:>10.4f}")

        title = f"{name} — Two Moons\n({params})"
        _plot_clusters(ax, X, labels, title)
        fig, single_ax = plt.subplots(figsize=(6, 4))
        _plot_clusters(single_ax, X, labels, title)
        fig.tight_layout()
        filename = name.lower().replace("-", "")
        fig.savefig(FIGURES_DIR / f"two_moons_{filename}.png", dpi=150)

    comparison.tight_layout()
    comparison.savefig(FIGURES_DIR / "two_moons_comparison.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
