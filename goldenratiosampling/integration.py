import numpy as np
from typing import Callable

from sampling import golden_sample, cartesian_sample

# NUMERICAL INTEGRATION


def golden_integral(func: Callable,
                    n_dim: int,
                    max_sample: int, *,
                    p: int = 1) -> list[float]:
    points = golden_sample(max_sample, n_dim)
    values = func(points)
    sums = np.cumsum(values)
    estimates = sums / np.arange(1.0, max_sample + 1.0)
    return estimates


def random_integral(func: Callable,
                    n_dim: int,
                    max_sample: int, *,
                    p: int = 1) -> list[float]:
    points = np.random.rand(max_sample, n_dim)
    values = func(points)
    sums = np.cumsum(values)
    estimates = sums / np.arange(1.0, max_sample + 1.0)
    return estimates


def cartesian_integral(func: Callable,
                       n_dim: int,
                       max_sample: int, *,
                       p: int = 1) -> (list[int], list[float]):
    n_points = np.zeros(max_sample)
    estimates = np.zeros(max_sample)
    for n in range(max_sample):
        points = cartesian_sample(n + 1, n_dim)
        n_points[n] = points.shape[0]
        values = func(points)
        sums = np.sum(values)
        estimates[n] = sums / n_points[n]
    return n_points, estimates
