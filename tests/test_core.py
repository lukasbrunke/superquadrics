import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from superquadrics.core import sign_pow, Superquadric, SuperquadricShape


def test_sign_pow_matches_signed_power():
    assert sign_pow(2.0, 2.0) == pytest.approx(4.0)
    assert sign_pow(-2.0, 2.0) == pytest.approx(-4.0)   # sign preserved
    assert sign_pow(0.0, 0.5) == pytest.approx(0.0)


def test_generate_mesh_shapes():
    resolution = 6
    sq = Superquadric(SuperquadricShape([1.0, 2.0, 3.0], [1.0, 1.0]), center=[0, 0, 0])
    vertices, triangles = sq.generate_mesh(resolution=resolution)
    assert vertices.shape == (resolution * resolution, 3)
    assert triangles.shape == (2 * (resolution - 1) ** 2, 3)
    # all triangle indices are valid vertex indices
    assert triangles.min() >= 0
    assert triangles.max() < vertices.shape[0]


def test_generate_mesh_unit_sphere_radius():
    # a=e=1 is a unit sphere: every local-frame vertex must have norm 1.
    sq = Superquadric(SuperquadricShape([1.0, 1.0, 1.0], [1.0, 1.0]), center=[0, 0, 0])
    vertices, _ = sq.generate_mesh(resolution=12)
    radii = np.linalg.norm(vertices, axis=1)
    np.testing.assert_allclose(radii, 1.0, atol=1e-9)


def test_to_mesh_returns_world_frame_on_surface():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [0.6, 0.9]), center=[0.5, -1.0, 2.0],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    vertices, triangles = sq.to_mesh(resolution=10)
    assert vertices.shape == (100, 3)
    assert triangles.shape == (2 * 81, 3)
    # world-frame vertices lie on the surface (F == 1); skip pole/seam rows
    interior = vertices.reshape(10, 10, 3)[1:-1, 1:-1].reshape(-1, 3)
    values = sq.inside_outside_function(interior.T)
    np.testing.assert_allclose(values, 1.0, atol=1e-6)


def _edge_face_counts(vertices, triangles):
    # Faces per undirected edge, keyed by rounded coordinates so the duplicated
    # seam/pole vertices collapse onto each other.
    from collections import Counter

    keys = [tuple(np.round(v, 9)) for v in vertices]
    counts = Counter()
    for tri in triangles:
        k = [keys[i] for i in tri]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            counts[frozenset((k[a], k[b]))] += 1
    return counts


def test_generate_render_mesh_watertight_without_degenerate_faces():
    # Small exponents are the hard case: the pole/seam snapping collapses faces there.
    sq = Superquadric(SuperquadricShape([0.5, 1.0, 1.0], [0.1, 0.1]))
    vertices, normals, triangles = sq.generate_render_mesh(resolution=10)
    assert vertices.shape == (100, 3)
    assert normals.shape == (100, 3)
    assert triangles.max() < vertices.shape[0]
    for tri in triangles:
        a, b, c = vertices[tri]
        assert np.linalg.norm(np.cross(b - a, c - a)) >= 1e-12   # no degenerate face
    # watertight: every (coordinate-keyed) edge borders exactly two faces
    assert all(count == 2 for count in _edge_face_counts(vertices, triangles).values())


def test_generate_render_mesh_normals_unit_outward_and_winding_consistent():
    sq = Superquadric(SuperquadricShape([0.5, 1.0, 1.5], [0.4, 0.8]))
    vertices, normals, triangles = sq.generate_render_mesh(resolution=12)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-9)
    # outward: positive dot with the radial direction (convex, origin-centred)
    radial = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    assert np.min(np.sum(normals * radial, axis=1)) > 0.0
    # face winding agrees with the vertex normals (CCW seen from outside)
    for tri in triangles:
        a, b, c = vertices[tri]
        assert np.dot(np.cross(b - a, c - a), normals[tri].mean(axis=0)) > 0.0


def test_generate_render_mesh_sphere_normals_are_radial():
    sq = Superquadric(SuperquadricShape([1.0, 1.0, 1.0], [1.0, 1.0]))
    vertices, normals, _ = sq.generate_render_mesh(resolution=12)
    radial = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    np.testing.assert_allclose(normals, radial, atol=1e-9)


def test_init_accepts_three_rotation_formats():
    eye = np.eye(3)
    sq_mat = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0], rotation=eye)
    sq_euler = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0], rotation=[0.0, 0.0, 0.0])
    sq_quat = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0], rotation=[0.0, 0.0, 0.0, 1.0])
    sq_none = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0])
    for sq in (sq_mat, sq_euler, sq_quat, sq_none):
        np.testing.assert_allclose(sq.rotation, eye, atol=1e-12)


def test_init_rejects_bad_rotation():
    with pytest.raises(ValueError):
        Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0], rotation=[1, 2, 3, 4, 5])


def test_transform_roundtrip_single_point():
    rot = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix()
    sq = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[1.0, 2.0, 3.0], rotation=rot)
    p_world = np.array([0.4, -0.2, 0.9])
    p_local = sq.transform_point_to_local(p_world)
    # transform_point_to_world expects a (3, N) column-stacked array
    back = sq.transform_point_to_world(p_local.reshape(3, 1)).ravel()
    np.testing.assert_allclose(back, p_world, atol=1e-12)


def test_shape_is_dataclass_backing_scales_and_exponents():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [0.6, 0.9]), center=[0, 0, 0])
    assert isinstance(sq.shape, SuperquadricShape)
    np.testing.assert_allclose(sq.shape.scales, [1.0, 1.5, 0.8])
    np.testing.assert_allclose(sq.shape.exponents, [0.6, 0.9])
    # scales/exponents are read-only views onto the shape
    np.testing.assert_allclose(sq.scales, sq.shape.scales)
    np.testing.assert_allclose(sq.exponents, sq.shape.exponents)
    with pytest.raises(AttributeError):
        sq.scales = [2.0, 2.0, 2.0]


def test_shape_dataclass_coerces_to_float_arrays():
    shape = SuperquadricShape([1, 2, 3], [1, 1])
    assert shape.scales.dtype == float
    assert shape.exponents.dtype == float


def test_shape_arrays_are_read_only():
    shape = SuperquadricShape([1.0, 2.0, 3.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        shape.scales[0] = 5.0
    with pytest.raises(ValueError):
        shape.exponents[0] = 5.0


def test_shape_copies_caller_arrays():
    scales = np.array([1.0, 2.0, 3.0])
    SuperquadricShape(scales, [1.0, 1.0])
    scales[0] = 9.0   # constructing a shape must not freeze the caller's array
    assert scales[0] == 9.0


def test_shape_rejects_nonpositive_scales():
    with pytest.raises(ValueError):
        SuperquadricShape([1.0, 0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        SuperquadricShape([1.0, -1.0, 1.0], [1.0, 1.0])


def test_shape_rejects_nonpositive_exponents():
    with pytest.raises(ValueError):
        SuperquadricShape([1.0, 1.0, 1.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        SuperquadricShape([1.0, 1.0, 1.0], [-0.5, 1.0])


def test_shape_rejects_wrong_lengths():
    with pytest.raises(ValueError):
        SuperquadricShape([1.0, 1.0], [1.0, 1.0])      # scales must be length 3
    with pytest.raises(ValueError):
        SuperquadricShape([1.0, 1.0, 1.0], [1.0])      # exponents must be length 2


def test_shape_warns_for_exponent_above_two():
    with pytest.warns(UserWarning):
        SuperquadricShape([1.0, 1.0, 1.0], [2.5, 1.0])


def test_shape_no_warning_at_exponent_two(recwarn):
    SuperquadricShape([1.0, 1.0, 1.0], [2.0, 2.0])     # boundary: e == 2 is allowed, no warning
    assert len(recwarn) == 0


def test_pose_setter_updates_center_and_rotation():
    sq = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0])
    rot = Rotation.from_euler("xyz", [0.1, 0.2, 0.3]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [1.0, 2.0, 3.0]
    sq.pose = T
    np.testing.assert_allclose(sq.rotation, rot, atol=1e-12)
    np.testing.assert_allclose(sq.center, [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(sq.pose, T, atol=1e-12)


def test_pose_setter_rejects_non_4x4():
    sq = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0])
    with pytest.raises(ValueError):
        sq.pose = np.eye(3)


def test_pose_property_and_inverse():
    rot = Rotation.from_euler("xyz", [0.3, -0.5, 1.1]).as_matrix()
    sq = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[1.0, 2.0, 3.0], rotation=rot)
    T = sq.pose
    assert T.shape == (4, 4)
    np.testing.assert_allclose(T[:3, :3], rot, atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0], atol=1e-12)
    np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)
    # pose_inverse undoes pose
    np.testing.assert_allclose(sq.pose @ sq.pose_inverse, np.eye(4), atol=1e-12)
    # pose_inverse matches transform_point_to_local on a sample point
    p = np.array([0.4, -0.2, 0.9])
    local_via_pose = (sq.pose_inverse @ np.append(p, 1.0))[:3]
    np.testing.assert_allclose(local_via_pose, sq.transform_point_to_local(p), atol=1e-12)


def test_transform_to_world_rejects_single_vector():
    sq = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0])
    with pytest.raises(ValueError):
        sq.transform_point_to_world(np.array([1.0, 2.0, 3.0]))  # (3,) not allowed; use reshape(3,1)


def test_inside_outside_classifies_centre_and_far_point():
    sq = Superquadric(SuperquadricShape([1, 2, 3], [0.7, 0.9]), center=[0, 0, 0], rotation=None)
    assert sq.inside_outside_function(np.array([0.0, 0.0, 0.0])) < 1.0   # centre is inside
    assert sq.inside_outside_function(np.array([10.0, 10.0, 10.0])) > 1.0  # far is outside


def test_surface_points_lie_on_surface():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [0.6, 0.9]), center=[0.5, -1.0, 2.0],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    x, y, z = sq.get_surface_points(n_points=25)
    # sample several interior grid points (avoid poles/seams at the borders)
    for (i, j) in [(7, 7), (12, 5), (5, 18), (18, 12)]:
        p = np.array([x[i, j], y[i, j], z[i, j]])
        assert sq.inside_outside_function(p) == pytest.approx(1.0, abs=1e-6)


def test_inside_outside_batch_matches_single():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [0.6, 0.9]), center=[0.5, -1.0, 2.0],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    pts = np.array([[0.1, 0.2, 0.3], [1.0, -0.5, 0.4], [2.0, 2.0, 2.0]])  # (N, 3)
    batch = sq.inside_outside_function(pts.T)                  # (3, N) -> (N,)
    single = np.array([sq.inside_outside_function(p) for p in pts])
    np.testing.assert_allclose(batch, single, rtol=1e-12, atol=1e-12)


def test_surface_points_uniform_mode_on_surface():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [0.6, 0.9]), center=[0.0, 0.0, 0.0])
    x, y, z = sq.get_surface_points(n_points=25, mode="uniform", n_fine=300)
    for (i, j) in [(7, 7), (12, 5), (5, 18), (18, 12)]:
        p = np.array([x[i, j], y[i, j], z[i, j]])
        assert sq.inside_outside_function(p) == pytest.approx(1.0, abs=1e-6)


def test_get_surface_points_rejects_bad_mode():
    sq = Superquadric(SuperquadricShape([1, 1, 1], [1, 1]), center=[0, 0, 0])
    with pytest.raises(ValueError):
        sq.get_surface_points(mode="bogus")


def test_small_exponents_close_poles_and_seam():
    # Regression: small exponents previously opened holes at the poles/seam,
    # because float ~1e-16 trig values do not vanish under a small power.
    for e in (0.5, 0.1, 0.01):
        sq = Superquadric(SuperquadricShape([1, 1, 1], [e, e]))

        # Mesh poles must collapse onto the z-axis (xy radius ~ 0).
        V, _ = sq.generate_mesh(resolution=40)
        zmax = np.abs(V[:, 2]).max()
        poles = V[np.abs(np.abs(V[:, 2]) - zmax) < 1e-9]
        assert np.linalg.norm(poles[:, :2], axis=1).max() < 1e-9

        # Surface seam (omega = -pi and +pi) must coincide.
        x, y, z = sq.get_surface_points(n_points=40)
        np.testing.assert_allclose(
            np.stack([x[0], y[0], z[0]]),
            np.stack([x[-1], y[-1], z[-1]]), atol=1e-9)


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
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [0.6, 0.9]), center=[0.2, -0.4, 0.1],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])  # off-axis, finite
    analytic = sq.grad_inside_outside_wrt_point(p)
    numeric = _fd_gradient(sq.inside_outside_function, p)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-5)


def test_gradient_finite_at_axis_aligned_point():
    # exponents <= 2: the gradient at an axis plane is finite (removable singularity).
    # Ellipsoid (e=1) at a point with y=0 has the exact gradient 2 P (p - c).
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [1.0, 1.0]), center=[0, 0, 0])
    p = np.array([1.0, 0.0, 0.5])  # y = 0 exactly
    g = sq.grad_inside_outside_wrt_point(p)
    assert np.all(np.isfinite(g))
    expected = 2.0 * _ellipsoid_PD(sq) @ (p - sq.center)
    np.testing.assert_allclose(g, expected, atol=1e-6)


def test_hessian_finite_at_axis_aligned_point():
    # Ellipsoid Hessian is the constant 2 P everywhere, including axis planes.
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [1.0, 1.0]), center=[0, 0, 0])
    p = np.array([1.0, 0.0, 0.5])  # y = 0 exactly
    H = sq.hessian_inside_outside_wrt_point(p)
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(H, 2.0 * _ellipsoid_PD(sq), atol=1e-6)


def test_gradient_equals_ellipsoid_when_exponents_one():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [1.0, 1.0]), center=[0.2, -0.4, 0.1],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])
    P = _ellipsoid_PD(sq)
    expected = 2.0 * P @ (p - sq.center)   # grad of (p-c)^T P (p-c)
    np.testing.assert_allclose(sq.grad_inside_outside_wrt_point(p), expected, rtol=1e-6, atol=1e-9)


def test_inside_outside_equals_ellipsoid_when_exponents_one():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [1.0, 1.0]), center=[0.2, -0.4, 0.1],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])
    d = p - sq.center
    expected = d @ _ellipsoid_PD(sq) @ d   # (p-c)^T P (p-c)
    assert sq.inside_outside_function(p) == pytest.approx(expected, rel=1e-9, abs=1e-12)


def _fd_hessian(grad_func, p, eps=1e-6):
    H = np.zeros((3, 3))
    for i in range(3):
        pp, pm = p.copy(), p.copy()
        pp[i] += eps
        pm[i] -= eps
        H[:, i] = (grad_func(pp) - grad_func(pm)) / (2 * eps)
    return 0.5 * (H + H.T)


def test_hessian_matches_finite_differences():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [0.6, 0.9]), center=[0.2, -0.4, 0.1],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])
    analytic = sq.hessian_inside_outside_wrt_point(p)
    numeric = _fd_hessian(sq.grad_inside_outside_wrt_point, p)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-3, atol=1e-4)


def test_hessian_equals_ellipsoid_when_exponents_one():
    sq = Superquadric(SuperquadricShape([1.0, 1.5, 0.8], [1.0, 1.0]), center=[0.2, -0.4, 0.1],
                      rotation=Rotation.from_euler("xyz", [0.2, 0.4, -0.3]).as_matrix())
    p = np.array([0.7, -0.9, 1.3])
    expected = 2.0 * _ellipsoid_PD(sq)   # Hessian of (p-c)^T P (p-c) is 2P (constant)
    np.testing.assert_allclose(sq.hessian_inside_outside_wrt_point(p), expected, rtol=1e-6, atol=1e-9)


def test_public_api_exports():
    import superquadrics

    expected = {
        "Superquadric",
        "SuperquadricShape",
        "sign_pow",
        "plot_quadric_open3d",
        "plot_quadric_pyvista",
        "superquadric_plotter",
    }
    assert set(superquadrics.__all__) == expected
    for name in expected:
        assert hasattr(superquadrics, name)
