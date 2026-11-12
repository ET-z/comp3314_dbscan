"""Baseline clustering tests. Run with: python test_clustering.py"""

import unittest

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from clarans import clarans
from kmeans import kmeans


class ClusteringTests(unittest.TestCase):
    def test_separated_clusters(self):
        """Both baselines should recover three well-separated blobs."""
        X, y = make_blobs(n_samples=90, centers=[[-8, 0], [0, 8], [8, 0]],
                          cluster_std=0.25, random_state=42)
        for algorithm in (kmeans, clarans):
            with self.subTest(algorithm=algorithm.__name__):
                labels = algorithm(X.tolist(), n_clusters=3)
                self.assertEqual(labels.shape, (len(X),))
                self.assertTrue(np.issubdtype(labels.dtype, np.integer))
                self.assertEqual(set(labels), {0, 1, 2})
                self.assertEqual(adjusted_rand_score(y, labels), 1.0)

    def test_reproducible_labels(self):
        X, _ = make_blobs(n_samples=60, centers=3, random_state=7)
        for algorithm in (kmeans, clarans):
            with self.subTest(algorithm=algorithm.__name__):
                first = algorithm(X, n_clusters=3, random_state=7)
                second = algorithm(X, n_clusters=3, random_state=7)
                np.testing.assert_array_equal(first, second)

    def test_clarans_duplicate_points(self):
        """Duplicate medoids must retain their own cluster on distance ties."""
        labels = clarans(np.zeros((8, 2)), n_clusters=3)
        self.assertEqual(set(labels), {0, 1, 2})

    def test_clarans_cluster_limits(self):
        X = np.array([[0.], [1.], [10.]])
        np.testing.assert_array_equal(clarans(X, n_clusters=1), [0, 0, 0])
        np.testing.assert_array_equal(clarans(X, n_clusters=3), [0, 1, 2])
        np.testing.assert_array_equal(clarans([[1., 2.]], n_clusters=1), [0])

    def test_clarans_invalid_parameters(self):
        X = np.array([[0.], [1.], [10.]])
        for name in ("n_clusters", "num_local", "max_neighbors"):
            for value in (0, -1, 1.5, True):
                with self.subTest(parameter=name, value=value):
                    params = {"n_clusters": 2, name: value}
                    with self.assertRaises(ValueError):
                        clarans(X, **params)
        with self.assertRaises(ValueError):
            clarans(X, n_clusters=4)
        for invalid in (np.empty((0, 2)), [[np.nan]], [[np.inf]], [0., 1.]):
            with self.subTest(data=invalid):
                with self.assertRaises(ValueError):
                    clarans(invalid, n_clusters=1)


if __name__ == "__main__":
    unittest.main()
