import numpy as np


def cartesian_product(*arrays):
    meshgrid = np.meshgrid(*arrays, indexing='ij')
    return np.dstack(meshgrid).reshape(-1, len(arrays))


def cartesian_sample(n_sample: int, n_dim: int):
    n_side = np.floor(np.power(n_sample, 1 / n_dim)).astype(int)
    array = np.linspace(0, 1, n_side, endpoint=False) + 0.5 / n_side
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
