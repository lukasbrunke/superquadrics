"""Superquadric geometry: the canonical Superquadric class and mesh helpers.

Pure geometry only — depends on numpy and scipy, never on visualization libraries.
Convention: a superquadric is described by ``center`` (world position), ``scales``
``[a, b, c]``, shape ``exponents`` ``[e1, e2]``, and a ``rotation`` (3x3 matrix,
Euler 'xyz' angles, or [x, y, z, w] quaternion).
"""

import numpy as np
from scipy.spatial.transform import Rotation
from scipy import interpolate


def sign_pow(x, a):
    """Compute sign(x) * |x|**a (the signed power used by superquadric surfaces)."""
    return np.sign(x) * (np.abs(x) ** a)


def generate_superquadric_mesh(a1, a2, a3, e1, e2, resolution=10):
    """Generate a triangle mesh for a canonical (origin-centred, axis-aligned) superquadric.

    Parameters
    ----------
    a1, a2, a3 : float
        Scale parameters along x, y, z.
    e1, e2 : float
        Shape exponents (north-south, east-west).
    resolution : int
        Number of samples along each parameter.

    Returns
    -------
    vertices : list[list[float]]
        ``resolution**2`` points in local coordinates.
    triangles : list[list[int]]
        ``2 * (resolution - 1)**2`` triangles as vertex-index triples.
    """
    eta = np.linspace(-np.pi / 2, np.pi / 2, resolution)
    omega = np.linspace(-np.pi, np.pi, resolution)
    eta, omega = np.meshgrid(eta, omega)

    x = a1 * sign_pow(np.cos(eta), e1) * sign_pow(np.cos(omega), e2)
    y = a2 * sign_pow(np.cos(eta), e1) * sign_pow(np.sin(omega), e2)
    z = a3 * sign_pow(np.sin(eta), e1)

    vertices = []
    for i in range(resolution):
        for j in range(resolution):
            vertices.append([x[i, j], y[i, j], z[i, j]])

    triangles = []
    for i in range(resolution - 1):
        for j in range(resolution - 1):
            v0 = i * resolution + j
            v1 = i * resolution + (j + 1)
            v2 = (i + 1) * resolution + j
            v3 = (i + 1) * resolution + (j + 1)
            triangles.append([v0, v1, v2])
            triangles.append([v1, v3, v2])

    return vertices, triangles
