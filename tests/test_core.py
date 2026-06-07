import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from superquadrics.core import sign_pow, generate_superquadric_mesh, Superquadric


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


def test_transform_to_world_rejects_single_vector():
    sq = Superquadric([0, 0, 0], [1, 1, 1], [1, 1])
    with pytest.raises(ValueError):
        sq.transform_point_to_world(np.array([1.0, 2.0, 3.0]))  # (3,) not allowed; use reshape(3,1)


def test_inside_outside_classifies_centre_and_far_point():
    sq = Superquadric([0, 0, 0], [1, 2, 3], [0.7, 0.9], rotation=None)
    assert sq.inside_outside_function(np.array([0.0, 0.0, 0.0])) < 1.0   # centre is inside
    assert sq.inside_outside_function(np.array([10.0, 10.0, 10.0])) > 1.0  # far is outside


def test_surface_points_lie_on_surface():
    sq = Superquadric([0.5, -1.0, 2.0], [1.0, 1.5, 0.8], [0.6, 0.9],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    x, y, z = sq.get_surface_points(n_points=25)
    # sample several interior grid points (avoid poles/seams at the borders)
    for (i, j) in [(7, 7), (12, 5), (5, 18), (18, 12)]:
        p = np.array([x[i, j], y[i, j], z[i, j]])
        assert sq.inside_outside_function(p) == pytest.approx(1.0, abs=1e-6)


def test_inside_outside_batch_matches_single():
    sq = Superquadric([0.5, -1.0, 2.0], [1.0, 1.5, 0.8], [0.6, 0.9],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    pts = np.array([[0.1, 0.2, 0.3], [1.0, -0.5, 0.4], [2.0, 2.0, 2.0]])  # (N, 3)
    batch = sq.inside_outside_function(pts.T)                  # (3, N) -> (N,)
    single = np.array([sq.inside_outside_function(p) for p in pts])
    np.testing.assert_allclose(batch, single, rtol=1e-12, atol=1e-12)


def test_surface_points_uniform_mode_on_surface():
    sq = Superquadric([0.0, 0.0, 0.0], [1.0, 1.5, 0.8], [0.6, 0.9])
    x, y, z = sq.get_surface_points(n_points=25, mode="uniform")
    for (i, j) in [(7, 7), (12, 5), (5, 18), (18, 12)]:
        p = np.array([x[i, j], y[i, j], z[i, j]])
        assert sq.inside_outside_function(p) == pytest.approx(1.0, abs=1e-6)


def test_get_surface_points_rejects_bad_mode():
    sq = Superquadric([0, 0, 0], [1, 1, 1], [1, 1])
    with pytest.raises(ValueError):
        sq.get_surface_points(mode="bogus")


def _fd_gradient(func, p, eps=1e-6):
    g = np.zeros(3)
    for i in range(3):
        pp, pm = p.copy(), p.copy()
        pp[i] += eps
        pm[i] -= eps
        g[i] = (func(pp) - func(pm)) / (2 * eps)
    return g


def _ellipsoid_PD(sq):
    # P = R diag(1/a^2,1/b^2,1/c^2) R^T  for the exponents=[1,1] ellipsoid
    D = np.diag(1.0 / sq.scales ** 2)
    return sq.rotation @ D @ sq.rotation.T


def test_gradient_matches_finite_differences():
    sq = Superquadric([0.2, -0.4, 0.1], [1.0, 1.5, 0.8], [0.6, 0.9],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])  # off-axis, finite
    analytic = sq.grad_inside_outside_wrt_point(p)
    numeric = _fd_gradient(sq.inside_outside_function, p)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_gradient_equals_ellipsoid_when_exponents_one():
    sq = Superquadric([0.2, -0.4, 0.1], [1.0, 1.5, 0.8], [1.0, 1.0],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])
    P = _ellipsoid_PD(sq)
    expected = 2.0 * P @ (p - sq.center)   # grad of (p-c)^T P (p-c)
    np.testing.assert_allclose(sq.grad_inside_outside_wrt_point(p), expected, rtol=1e-6, atol=1e-9)


def _fd_hessian(grad_func, p, eps=1e-6):
    H = np.zeros((3, 3))
    for i in range(3):
        pp, pm = p.copy(), p.copy()
        pp[i] += eps
        pm[i] -= eps
        H[:, i] = (grad_func(pp) - grad_func(pm)) / (2 * eps)
    return 0.5 * (H + H.T)


def test_hessian_matches_finite_differences():
    sq = Superquadric([0.2, -0.4, 0.1], [1.0, 1.5, 0.8], [0.6, 0.9],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])
    analytic = sq.hessian_inside_outside_wrt_point(p)
    numeric = _fd_hessian(sq.grad_inside_outside_wrt_point, p)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-4)


def test_hessian_equals_ellipsoid_when_exponents_one():
    sq = Superquadric([0.2, -0.4, 0.1], [1.0, 1.5, 0.8], [1.0, 1.0],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])
    expected = 2.0 * _ellipsoid_PD(sq)   # Hessian of (p-c)^T P (p-c) is 2P (constant)
    np.testing.assert_allclose(sq.hessian_inside_outside_wrt_point(p), expected, rtol=1e-6, atol=1e-9)
