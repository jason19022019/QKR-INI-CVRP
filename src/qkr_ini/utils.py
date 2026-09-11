"""Small shared helpers used across multiple pipeline stages."""
import numpy as np


def eucl_dist(a, b):
    """Euclidean distance between two coordinate arrays/points."""
    return np.sqrt(np.sum((a - b) ** 2))
