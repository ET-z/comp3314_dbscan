
import numpy as np


class DBSCAN:
    def __init__(self, eps=0.5, min_pts=5):
        self.eps = eps
        self.min_pts = min_pts

        self.labels_ = None

    def region_query(self, X, point_index):
        """
        Find all points within eps distance of a point.

        TODO: Implement epsilon-neighborhood search.
        """
        pass

    def fit(self, X):
        """
        Perform DBSCAN clustering.

        TODO: Implement clustering algorithm.
        """
        pass
