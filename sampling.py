import numpy as np
from sympy.solvers import nsolve
from sympy import Symbol

from cartesian import spherical_to_cartesian


def golden_vector(n_dim: int, *, p: int = 1) -> list[float]:
    """ Returns N-dimensional golden vector.

    Generalisation of the ratio a n dimensional number or vector as defined
    by Anderson (1993) [1].

    Arguments:
        n_dim: interger
            Dimension of golden vector. Example: n_dim = 1 (and p=1)
            corresponds to golden ratio phi  =  0.618 .
        p: integer (Default: 1)
            Period of golden vector. p multiples of the golden vector fit
            within the interval [0,1]^n_dim. Example: for n_dim = 1, the
            golden vectors corresponding to p = 2, 3 are also known
            as the silver, bronze, etc ratios.
    Returns:
        vec: Vector

    [1] P.G. Anderson (1993), 'Multidimensional Golden Means'
        in 'Application of Fibonacci numbers'.
    """
    assert isinstance(n_dim, int) and n_dim > 0
    assert isinstance(p, int) and p > 0
    vec = np.zeros(n_dim)
    phi = Symbol('phi')
    vec[0] = nsolve(phi * (phi + p) ** n_dim - 1, 1)
    for k in range(1, n_dim):
        vec[k] = vec[0] * (vec[0] + p) ** k
    return vec


def golden_sample(n_sample: int, n_dim: int, *, p: int = 1) -> list[list[float]]:
    """Returns a sample generated from the N-dimensional golden mean.

    Sample of fixed length n_sample within the (Multidimensional)
    interval [0,1]^n_dim.

    Arguments:
        n_sample: integer
            Sample size

    Returns:
        Sample: List[Vector]
            Sample of n_sample integer multiple of the n_dim golden vector
            restricted to the (Multidimensional) interval [0,1]^n_dim.
    """
    vec = golden_vector(n_dim, p=p)
    return (n * vec % 1 for n in range(n_sample))


# TODO: (re)write generator variant
def golden_sphere_sampling(n_sample: int,
                           n_dim: int, *,
                           p: int = 1,
                           cartesian_output: bool = True) -> list[list[float]]:
    """ Returns golden sample on N-dimensional unit sphere.

    Golden sample on S^n_dim, surface of D^(n_dim+1)
    """
    point_coordinates = golden_sample(n_sample, n_dim, p=p).T
    radii = np.ones(n_sample)
    angles = np.arccos(2 * point_coordinates[:-1] - 1)
    last_angle = 2 * np.pi * point_coordinates[-1]
    spherical_coordinates = np.array([radii, *angles, last_angle])
    points = spherical_coordinates.T
    if cartesian_output:
        points = [spherical_to_cartesian(p) for p in points]
    return np.array(points)


# TODO: (re)write generator variant
def golden_ball_sampling(n_sample: int,
                         n_dim: int, *,
                         p: int = 1,
                         cartesian_output: bool = True) -> list[list[float]]:
    """Returns golden sampling in N-dimensional unit ball.
    Golden sample on the ball D^n_dim.
    """
    point_coordinates = golden_sample(n_sample, n_dim, p=p).T
    radii = np.power(point_coordinates[0], 1 / n_dim)
    angles = np.arccos(2 * point_coordinates[1:-1] - 1)
    last_angle = 2 * np.pi * point_coordinates[-1]
    spherical_coordinates = np.array([radii, *angles, last_angle])
    points = spherical_coordinates.T
    if cartesian_output:
        points = [spherical_to_cartesian(p) for p in points]
    return np.array(points)


# TODO: (re)write generator variant
def golden_radial_sampling(n_sample: int,
                           n_dim: int, *,
                           p: int = 1,
                           cartesian: bool = True,
                           radial_spacing: float = 1.0) -> list[list[float]]:
    '''

    '''
    points = golden_sphere_sampling(n_sample, n_dim - 1, p=p, cartesian=False)
    r = np.arange(n_sample) * radial_spacing
    r = np.power(r, 1 / n_dim)
    points[:, 0] *= r
    if cartesian:
        points = [spherical_to_cartesian(p) for p in points]
    return np.array(points)
