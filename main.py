
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import make_moons


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
    plt.show()


if __name__ == "__main__":
    main()
