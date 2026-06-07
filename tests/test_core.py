import numpy as np
import pytest

from superquadrics.core import sign_pow, generate_superquadric_mesh


def test_sign_pow_matches_signed_power():
    assert sign_pow(2.0, 2.0) == pytest.approx(4.0)
    assert sign_pow(-2.0, 2.0) == pytest.approx(-4.0)   # sign preserved
    assert sign_pow(0.0, 0.5) == pytest.approx(0.0)


def test_generate_superquadric_mesh_shapes():
    resolution = 6
    vertices, triangles = generate_superquadric_mesh(1.0, 2.0, 3.0, 1.0, 1.0, resolution=resolution)
    vertices = np.asarray(vertices)
    triangles = np.asarray(triangles)
    assert vertices.shape == (resolution * resolution, 3)
    assert triangles.shape == (2 * (resolution - 1) ** 2, 3)
    # all triangle indices are valid vertex indices
    assert triangles.min() >= 0
    assert triangles.max() < vertices.shape[0]


def test_generate_superquadric_mesh_unit_sphere_radius():
    # a=e=1 is a unit sphere: every surface vertex must have norm 1.
    vertices, _ = generate_superquadric_mesh(1.0, 1.0, 1.0, 1.0, 1.0, resolution=12)
    radii = np.linalg.norm(np.asarray(vertices), axis=1)
    np.testing.assert_allclose(radii, 1.0, atol=1e-9)
