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


from superquadrics.core import Superquadric
from scipy.spatial.transform import Rotation


def test_init_accepts_three_rotation_formats():
    eye = np.eye(3)
    sq_mat = Superquadric([0, 0, 0], [1, 1, 1], [1, 1], rotation=eye)
    sq_euler = Superquadric([0, 0, 0], [1, 1, 1], [1, 1], rotation=[0.0, 0.0, 0.0])
    sq_quat = Superquadric([0, 0, 0], [1, 1, 1], [1, 1], rotation=[0.0, 0.0, 0.0, 1.0])
    sq_none = Superquadric([0, 0, 0], [1, 1, 1], [1, 1])
    for sq in (sq_mat, sq_euler, sq_quat, sq_none):
        np.testing.assert_allclose(sq.rotation, eye, atol=1e-12)


def test_init_rejects_bad_rotation():
    with pytest.raises(ValueError):
        Superquadric([0, 0, 0], [1, 1, 1], [1, 1], rotation=[1, 2, 3, 4, 5])


def test_transform_roundtrip_single_point():
    rot = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix()
    sq = Superquadric([1.0, 2.0, 3.0], [1, 1, 1], [1, 1], rotation=rot)
    p_world = np.array([0.4, -0.2, 0.9])
    p_local = sq.transform_point_to_local(p_world)
    # transform_point_to_world expects a (3, N) column-stacked array
    back = sq.transform_point_to_world(p_local.reshape(3, 1)).ravel()
    np.testing.assert_allclose(back, p_world, atol=1e-12)
