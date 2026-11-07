
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from dbscan import NOISE, dbscan


FIGURES_DIR = Path(__file__).parent / "figures"


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    # Generate a simple 2D clustering dataset
    X, _ = make_moons(
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

    eps, min_pts = 0.2, 4
    labels = dbscan(X, eps=eps, min_pts=min_pts)
    n_clusters = len(set(labels) - {NOISE})
    n_noise = int((labels == NOISE).sum())
    print(f"Clusters found: {n_clusters}, noise points: {n_noise}")

    plt.figure(figsize=(6, 4))
    clustered = labels != NOISE
    plt.scatter(X[clustered, 0], X[clustered, 1],
                c=labels[clustered], cmap="tab10", s=15)
    if n_noise:
        plt.scatter(X[~clustered, 0], X[~clustered, 1],
                    c="gray", marker="x", s=15, label="Noise")
        plt.legend()
    plt.title(f"DBSCAN — Two Moons (eps={eps}, MinPts={min_pts})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "two_moons_dbscan.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
