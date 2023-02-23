import numpy as np
import matplotlib.pyplot as plt
from sympy.solvers import nsolve
from sympy import Symbol


# GOLDEN RATIO SAMPLING FUNCTIONS

def golden_vec(n_dim: int, *, p: int = 1):
    assert isinstance(n_dim, int) and n_dim > 0
    assert isinstance(p, int) and p > 0
    vec = np.zeros(n_dim)
    phi = Symbol('phi')
    vec[0] = nsolve(phi * (phi + p) ** n_dim - 1, 1)
    for k in range(1, n_dim):
        vec[k] = vec[0] * (vec[0] + p) ** k
    return vec


def golden_sample(n_sample: int, n_dim: int, *, p: int = 1):
    vec = golden_vec(n_dim, p=p)
    return np.array([n * vec % 1 for n in range(n_sample)])


def golden_sphere_sampling(n_sample: int, n_dim: int, *, p: int = 1, cartesian: bool = True):
    point_coordinates = golden_sample(n_sample, n_dim, p=p).T
    angles = np.arccos(2 * point_coordinates[:-1] - 1)
    last_angle = 2 * np.pi * point_coordinates[-1]
    spherical_coordinates = np.array([np.ones(n_sample), *angles, last_angle])
    points = spherical_coordinates.T
    if cartesian:
        points = [spherical_to_cartesian(p) for p in points]
    return np.array(points)


def golden_ball_sampling(n_sample: int, n_dim: int, *, p: int = 1, cartesian: bool = True):
    points = golden_sphere_sampling(n_sample, n_dim - 1, p=p, cartesian=False)
    r = np.linspace(0, 1, n_sample)
    r = np.power(r, 1 / n_dim)
    points[:, 0] *= r
    if cartesian:
        points = [spherical_to_cartesian(p) for p in points]
    return np.array(points)


# CARTESIAN FUNCTIONS

def cartesian_product(*arrays):
    meshgrid = np.meshgrid(*arrays, indexing='ij')
    return np.dstack(meshgrid).reshape(-1, len(arrays))


def cartesian_sample(n_sample: int, n_dim: int, *, endpoint: bool = False):
    n_side = np.floor(np.power(n_sample, 1 / n_dim)).astype(int)
    array = np.linspace(0, 1, n_side, endpoint=endpoint)
    arrays = [array] * n_dim
    return cartesian_product(*arrays)


def cartesian_to_spherical(x):
    n_dim = x.shape
    assert len(n_dim) == 1
    y = np.zeros(n_dim)
    y[0] = np.sqrt(x.dot(x))
    for i in range(1, n_dim):
        z = x[i-1:]
        y[i] = x[i-1] / np.sqrt(z.dot(z))
    if x[-1] < 0:
        y[-1] = 2*np.pi - y[-1]
    return y


def spherical_to_cartesian(x):
    dims = x.shape
    n_dim = dims[0]
    r = x[0]
    sin = np.sin(x[1:])
    cos = np.cos(x[1:])
    y = np.zeros(n_dim)
    for i in range(n_dim - 1):
        y[i] = r * cos[i]
        for j in range(i):
            y[i] *= sin[j]
    y[-1] = r
    for j in range(n_dim - 1):
        y[-1] *= sin[j]
    return y


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
