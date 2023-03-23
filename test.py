import numpy as np
import matplotlib.pyplot as plt

from sampling import golden_sample, cartesian_sample, golden_ball_sampling, golden_sphere_sampling
from integration import golden_integral, cartesian_integral, random_integral

# PI SAMPLING FUNCTIONS


def pi_sample_cumulative(points):
    n_points = points.shape[0]
    radii = np.sqrt((points*points).sum(axis=1))
    inside = (radii < 1).astype(float)
    hits = np.cumsum(inside)
    totals = np.arange(1.0, n_points + 1.0)
    ratios = hits / totals
    estimates = ratios * 4.0
    return estimates


def pi_sample(points):
    n_points = points.shape[0]
    radii = np.sqrt((points*points).sum(axis=1))
    inside = (radii < 1).astype(float)
    hits = np.sum(inside)
    ratio = hits / n_points
    estimate = ratio * 4.0
    return estimate


def pi_golden_sample(max_sample: int):
    points = golden_sample(max_sample, 2)
    estimates = pi_sample_cumulative(points)
    return estimates


def pi_random_sample(max_sample: int):
    points = np.random.rand(max_sample, 2)
    estimates = pi_sample_cumulative(points)
    return estimates


def pi_cartesian_sample(max_sample: int):
    n_points = np.zeros(max_sample)
    estimates = np.zeros(max_sample)
    for n in range(max_sample):
        points = cartesian_sample(n + 1, 2)
        n_points[n] = points.shape[0]
        estimates[n] = pi_sample(points)
    return n_points, estimates


# PLOTTING FUNCTIONS

def plot_gaussian_integral(n_sample: int, *, n_dim: int = 1, p: int = 1):
    def gaussian(x, sigma=1):
        return np.exp(-np.sum((x/sigma)**2, axis=-1))
    golden = golden_integral(gaussian, n_dim, n_sample)
    random = random_integral(gaussian, n_dim, n_sample)
    n_points, cartesian = cartesian_integral(gaussian, n_dim, n_sample)
    plt.plot(golden, label='golden sample')
    plt.plot(random, label='random sample')
    plt.plot(n_points, cartesian, label='cartesian sample')
    plt.xscale('log')
    plt.legend()
    plt.show()


def plot_golden_sample(n_sample: int, *, p: int = 1):
    x, y = golden_sample(n_sample, 2, p=p).T
    colors = np.linspace(0, 1, n_sample)

    plt.scatter(x, y, c=colors)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()


def plot_pi_sampling(n: int):
    golden = pi_golden_sample(n)
    random = pi_random_sample(n)
    n_points, cartesian = pi_cartesian_sample(n)
    plt.plot(golden, label='golden sample')
    plt.plot(random, label='random sample')
    plt.plot(n_points, cartesian, label='cartesian sample')
    plt.xscale('log')
    plt.axhline(np.pi)
    plt.legend()
    plt.show()


def plot_ball_sampling(n_sample: int):
    fig, ax = plt.subplots(2, 2)

    x, y = golden_ball_sampling(n_sample, 2).T
    colors = np.linspace(0, 1, n_sample)
    ax[0, 0].scatter(x, y, c=colors)
    ax[0, 0].set_xlim([-1, 1])
    ax[0, 0].set_ylim([-1, 1])

    x, y = golden_ball_sampling(n_sample, 2, p=2).T
    colors = np.linspace(0, 1, n_sample)
    ax[0, 1].scatter(x, y, c=colors)
    ax[0, 1].set_xlim([-1, 1])
    ax[0, 1].set_ylim([-1, 1])

    x, y = golden_ball_sampling(n_sample, 2, p=3).T
    colors = np.linspace(0, 1, n_sample)
    ax[1, 0].scatter(x, y, c=colors)
    ax[1, 0].set_xlim([-1, 1])
    ax[1, 0].set_ylim([-1, 1])

    x, y = golden_ball_sampling(n_sample, 2, p=4).T
    colors = np.linspace(0, 1, n_sample)
    ax[1, 1].scatter(x, y, c=colors)
    ax[1, 1].set_xlim([-1, 1])
    ax[1, 1].set_ylim([-1, 1])

    plt.show()


def plot_sphere_sampling(n_sample: int):
    x, y, z = golden_sphere_sampling(n_sample, 2).T
    colors = np.linspace(0, 1, n_sample)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x, y, z, c=colors)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    plt.show()
