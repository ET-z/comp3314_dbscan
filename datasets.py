"""Small examples inspired by the paper, not its original point coordinates."""

import numpy as np


def make_datasets(seed=42):
    rng = np.random.default_rng(seed)

    def disk(center, radius, n):
        angle = rng.uniform(0, 2 * np.pi, n)
        r = radius * np.sqrt(rng.uniform(0, 1, n))
        return np.column_stack((r * np.cos(angle), r * np.sin(angle))) + center

    def combine(groups):
        X = np.vstack(groups)
        y = np.concatenate([np.full(len(group), i) for i, group in enumerate(groups)])
        order = rng.permutation(len(X))
        return X[order], y[order]

    # Four round clusters, with one much larger than the others.
    compact = combine([
        disk([0, 0], 1.25, 64),
        disk([-2, -1.5], 0.45, 16),
        disk([2, 1.7], 0.4, 12),
        disk([2, -1.7], 0.4, 12),
    ])

    # Two pairs of interlocking moons, making four nonconvex clusters.
    t = np.linspace(0, np.pi, 40)
    top = np.column_stack((np.cos(t), np.sin(t)))
    bottom = np.column_stack((1 - np.cos(t), 0.5 - np.sin(t)))
    groups = []
    for offset in ([0, 0], [0, 3]):
        for arc in (top, bottom):
            groups.append(arc + offset + rng.normal(0, 0.02, arc.shape))
    nonconvex = combine(groups)

    # Different shapes and sizes, plus some scattered noise.
    t = np.linspace(0.1, np.pi - 0.1, 44)
    arc = np.column_stack((1 + 1.15 * np.cos(t), 1 + 1.15 * np.sin(t)))
    arc += rng.normal(0, 0.02, arc.shape)
    line = np.column_stack((np.linspace(0, 2, 36), rng.normal(-1.7, 0.06, 36)))
    X, y = combine([disk([-2, 1], 0.55, 36), arc,
                    line, disk([-2, -1.8], 0.3, 16)])
    noise = []
    # Keep the added outliers away from clusters and from each other.
    while len(noise) < 15:
        point = rng.uniform([-3.2, -2.8], [3, 3.1])
        if np.linalg.norm(X - point, axis=1).min() < 0.5:
            continue
        if noise and np.linalg.norm(np.array(noise) - point, axis=1).min() < 0.5:
            continue
        noise.append(point)
    X = np.vstack((X, noise))
    y = np.concatenate((y, np.full(len(noise), -1)))
    order = rng.permutation(len(X))
    noisy = X[order], y[order]

    # name, points, generated labels, chosen epsilon
    return [
        ("compact", *compact, 0.55),
        ("nonconvex", *nonconvex, 0.22),
        ("noisy", *noisy, 0.32),
    ]
