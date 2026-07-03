"""Superquadric geometry: the canonical Superquadric class and mesh helpers.

Pure geometry only — depends on numpy and scipy, never on visualization libraries.
Convention: a superquadric is described by ``center`` (world position), ``scales``
``[a, b, c]``, shape ``exponents`` ``[e1, e2]``, and a ``rotation`` (3x3 matrix,
Euler 'xyz' angles, or [x, y, z, w] quaternion).
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
from scipy import interpolate


# Relative (dimensionless) clamp applied to the *normalized* local coordinates
# x/a, y/b, z/c in the gradient/Hessian. At axis-aligned points those formulas
# divide by the normalized coordinate; clamping it off the axis plane keeps the
# result finite (a removable singularity for exponents <= 2). Because the clamp
# acts on the normalized coordinate it is automatically scale-aware in a, b, c.
_AXIS_EPS = 1e-9


def sign_pow(x, a):
    """Compute sign(x) * |x|**a (the signed power used by superquadric surfaces)."""
    return np.sign(x) * (np.abs(x) ** a)


# Trig values below this magnitude are snapped to exactly 0 before the power.
_TRIG_SNAP = 1e-9


def _snap_zero(values):
    """Snap near-zero values to exactly 0.

    The spherical-product parametrization evaluates cos/sin at the poles
    (eta = +-pi/2) and the seam (omega = +-pi), where they are mathematically 0
    but float error leaves ~1e-16. For small exponents ``|1e-16|**e`` is NOT
    small (e.g. ``1e-16 ** 0.01 ~= 0.7``), so the pole ring and seam fail to
    close and the mesh develops holes. Snapping to 0 keeps those vertices
    coincident for any exponent.
    """
    values = np.asarray(values, dtype=float)
    return np.where(np.abs(values) < _TRIG_SNAP, 0.0, values)


def _surface_xyz(scales, exponents, eta, omega):
    """Superquadric surface coordinates for parameter arrays ``eta`` (in
    [-pi/2, pi/2]) and ``omega`` (in [-pi, pi]), with pole/seam-safe snapping.
    Returns (x, y, z) arrays shaped like the inputs."""
    a1, a2, a3 = scales
    e1, e2 = exponents
    cos_eta = _snap_zero(np.cos(eta))
    sin_eta = _snap_zero(np.sin(eta))
    cos_omega = _snap_zero(np.cos(omega))
    sin_omega = _snap_zero(np.sin(omega))
    x = a1 * sign_pow(cos_eta, e1) * sign_pow(cos_omega, e2)
    y = a2 * sign_pow(cos_eta, e1) * sign_pow(sin_omega, e2)
    z = a3 * sign_pow(sin_eta, e1)
    return x, y, z


def _surface_normals(scales, exponents, eta, omega):
    """Unit outward surface normals for parameter arrays ``eta`` and ``omega``,
    with the same pole/seam-safe snapping as :func:`_surface_xyz`.

    The analytic normal of the superquadric surface is::

        n ∝ [ sign_pow(cos eta, 2-e1) * sign_pow(cos omega, 2-e2) / a1,
              sign_pow(cos eta, 2-e1) * sign_pow(sin omega, 2-e2) / a2,
              sign_pow(sin eta, 2-e1) / a3 ]

    which stays well-defined everywhere, including the poles and seam where the
    *triangulation* of the surface degenerates. Returns (nx, ny, nz) unit-normal
    component arrays shaped like the inputs.
    """
    a1, a2, a3 = scales
    e1, e2 = exponents
    cos_eta = _snap_zero(np.cos(eta))
    sin_eta = _snap_zero(np.sin(eta))
    cos_omega = _snap_zero(np.cos(omega))
    sin_omega = _snap_zero(np.sin(omega))
    nx = sign_pow(cos_eta, 2.0 - e1) * sign_pow(cos_omega, 2.0 - e2) / a1
    ny = sign_pow(cos_eta, 2.0 - e1) * sign_pow(sin_omega, 2.0 - e2) / a2
    nz = sign_pow(sin_eta, 2.0 - e1) / a3
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return nx / norm, ny / norm, nz / norm


def _clamp_off_axis(t):
    """Sign-preserving clamp of a normalized coordinate away from 0 by ``_AXIS_EPS``."""
    if abs(t) < _AXIS_EPS:
        return _AXIS_EPS if t >= 0 else -_AXIS_EPS
    return t


@dataclass(eq=False)
class SuperquadricShape:
    """Pose-free shape parameters of a superquadric.

    Attributes:
        scales: [a, b, c] scale parameters.
        exponents: [e1, e2] shape parameters.
    """

    scales: np.ndarray
    exponents: np.ndarray

    def __post_init__(self):
        # Copy (don't alias the caller's array) and make read-only so the shape
        # is effectively immutable, matching the read-only views on Superquadric.
        self.scales = np.array(self.scales, dtype=float)
        self.exponents = np.array(self.exponents, dtype=float)
        if self.scales.shape != (3,):
            raise ValueError("scales must have length 3 ([a, b, c])")
        if self.exponents.shape != (2,):
            raise ValueError("exponents must have length 2 ([e1, e2])")
        if np.any(self.scales <= 0):
            raise ValueError("scales must be positive")
        if np.any(self.exponents <= 0):
            raise ValueError("exponents must be positive")
        if np.any(self.exponents > 2):
            warnings.warn(
                "superquadric exponents > 2 give non-convex shapes; the inside-outside "
                "gradient/Hessian are singular at axis-aligned points (regularized)",
                stacklevel=2)
        self.scales.flags.writeable = False
        self.exponents.flags.writeable = False


def _to_rotation_matrix(rotation):
    """Coerce a rotation given as a 3x3 matrix, Euler 'xyz' angles, or [x,y,z,w]
    quaternion into a 3x3 rotation matrix. ``None`` yields the identity."""
    if rotation is None:
        return np.eye(3)
    if isinstance(rotation, (list, np.ndarray)):
        rotation = np.array(rotation, dtype=float)
        if rotation.shape == (3, 3):
            return rotation
        if rotation.shape == (3,):
            return Rotation.from_euler("xyz", rotation).as_matrix()
        if rotation.shape == (4,):
            return Rotation.from_quat(rotation).as_matrix()
    raise ValueError("Invalid rotation format")


class Superquadric:
    """A superquadric: a :class:`SuperquadricShape` placed at a pose.

    The defining shape parameters are supplied as a :class:`SuperquadricShape`
    (``self.shape``); ``scales`` and ``exponents`` are read-only views onto it.
    The pose (``center`` plus ``rotation``) is exposed as the settable
    :attr:`pose` property.
    """

    def __init__(self, shape, center=(0.0, 0.0, 0.0), rotation=None):
        """
        Args:
            shape: a :class:`SuperquadricShape` (scales + exponents).
            center: [x, y, z] world coordinates of the centre (default origin).
            rotation: one of a 3x3 matrix, Euler 'xyz' angles (len 3),
                or an [x, y, z, w] quaternion (len 4). Defaults to identity.
        """
        if not isinstance(shape, SuperquadricShape):
            raise TypeError("shape must be a SuperquadricShape")
        self.shape = shape
        self.center = np.array(center, dtype=float)
        self.rotation = _to_rotation_matrix(rotation)

    @property
    def scales(self):
        """[a, b, c] scale parameters (read-only view of ``self.shape``)."""
        return self.shape.scales

    @property
    def exponents(self):
        """[e1, e2] shape parameters (read-only view of ``self.shape``)."""
        return self.shape.exponents

    @property
    def pose(self):
        """4x4 homogeneous transform mapping the local (body) frame to the world."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.center
        return T

    @pose.setter
    def pose(self, transform):
        """Update center and rotation from a 4x4 homogeneous transform."""
        transform = np.asarray(transform, dtype=float)
        if transform.shape != (4, 4):
            raise ValueError("pose must be a 4x4 homogeneous transform")
        self.rotation = transform[:3, :3].copy()
        self.center = transform[:3, 3].copy()

    @property
    def pose_inverse(self):
        """4x4 homogeneous transform mapping the world frame to the local frame."""
        T = np.eye(4)
        T[:3, :3] = self.rotation.T
        T[:3, 3] = -self.rotation.T @ self.center
        return T

    def transform_point_to_local(self, point):
        """World -> local. Accepts a single (3,) point or a (3, N) array; the
        output mirrors the input shape. (Unlike :meth:`transform_point_to_world`,
        which is (3, N)-only, a single point need not be reshaped here.)
        """
        point = np.asarray(point)
        if point.shape[0] != 3:
            raise ValueError("point must have length 3 along its first axis")
        if point.ndim == 2:
            return np.dot(self.rotation.T, (point.T - self.center).T)
        return np.dot(self.rotation.T, point - self.center)

    def transform_point_to_world(self, point):
        """Local -> world. Expects a (3, N) column-stacked array of points.

        For a single point, pass it as ``point.reshape(3, 1)``.
        """
        point = np.asarray(point)
        if point.ndim != 2 or point.shape[0] != 3:
            raise ValueError("point must be a (3, N) array; for a single point use point.reshape(3, 1)")
        return np.dot(self.rotation, point) + self.center.reshape(-1, 1)

    def inside_outside_function(self, point):
        """Inside-outside value F. F < 1 inside, F = 1 on surface, F > 1 outside.

        Accepts a single (3,) world point (returns a scalar) or a (3, N) array
        of points (returns an (N,) array).
        """
        local_point = self.transform_point_to_local(point)
        scales = self.scales.reshape(-1, 1) if local_point.ndim == 2 else self.scales
        normalized = local_point / scales
        e1, e2 = self.exponents
        x, y, z = normalized
        term1 = (abs(x) ** (2 / e2) + abs(y) ** (2 / e2)) ** (e2 / e1)
        term2 = abs(z) ** (2 / e1)
        return term1 + term2

    def get_surface_points(self, n_points=20, scaling=None, n_v=None, n_u=None,
                           mode="simple", n_fine=1000):
        """Return (x, y, z) grids of world-frame surface points, each shaped (n_v, n_u).

        ``n_u`` samples the u parameter in [-pi/2, pi/2]; ``n_v`` samples v in
        [-pi, pi]; both default to ``n_points``. mode='simple' uses uniform angle
        sampling; mode='uniform' reparameterizes by arc length for more even
        spacing, using ``n_fine`` samples for the arc-length integration.
        ``scaling`` (if not None) inflates the surface by ``scaling ** (e1/2)``.
        """
        e1, e2 = self.exponents
        a1, a2, a3 = self.scales
        if n_v is None:
            n_v = n_points
        if n_u is None:
            n_u = n_points

        if mode == "simple":
            u = np.linspace(-np.pi / 2, np.pi / 2, n_u)
            v = np.linspace(-np.pi, np.pi, n_v)
            u, v = np.meshgrid(u, v)
            x, y, z = _surface_xyz(self.scales, self.exponents, u, v)
        elif mode == "uniform":
            u_fine = np.linspace(-np.pi / 2, np.pi / 2, n_fine)
            v_fine = np.linspace(-np.pi, np.pi, n_fine)

            # v sampling at u = 0 (xy-plane)
            x_v = a1 * sign_pow(np.cos(v_fine), e2)
            y_v = a2 * sign_pow(np.sin(v_fine), e2)
            z_v = np.zeros_like(v_fine)
            dv = np.sqrt(np.diff(x_v) ** 2 + np.diff(y_v) ** 2 + np.diff(z_v) ** 2)
            v_arclen = np.insert(np.cumsum(dv), 0, 0.0)
            v_arclen /= v_arclen[-1]
            v_vals = interpolate.interp1d(
                v_arclen, v_fine, kind="linear", bounds_error=False,
                fill_value=(v_fine[0], v_fine[-1]))(np.linspace(0, 1, n_v))

            # u sampling at v = 0 (xz-plane)
            x_u = a1 * sign_pow(np.cos(u_fine), e1)
            y_u = np.zeros_like(u_fine)  # v fixed at 0 -> y is identically zero along this curve
            z_u = a3 * sign_pow(np.sin(u_fine), e1)
            du = np.sqrt(np.diff(x_u) ** 2 + np.diff(y_u) ** 2 + np.diff(z_u) ** 2)
            u_arclen = np.insert(np.cumsum(du), 0, 0.0)
            u_arclen /= u_arclen[-1]
            u_vals = interpolate.interp1d(
                u_arclen, u_fine, kind="linear", bounds_error=False,
                fill_value=(u_fine[0], u_fine[-1]))(np.linspace(0, 1, n_u))

            u, v = np.meshgrid(u_vals, v_vals)
            x, y, z = _surface_xyz(self.scales, self.exponents, u, v)
        else:
            raise ValueError(f"Invalid mode: {mode!r}")

        if scaling is not None:
            scaling_factor = scaling ** (e1 / 2.0)
            x = x * scaling_factor
            y = y * scaling_factor
            z = z * scaling_factor

        local_points = np.stack([x.flatten(), y.flatten(), z.flatten()])
        world_points = self.transform_point_to_world(local_points)
        x = world_points[0].reshape(n_v, n_u)
        y = world_points[1].reshape(n_v, n_u)
        z = world_points[2].reshape(n_v, n_u)
        return x, y, z

    def generate_mesh(self, resolution=20):
        """Triangle mesh of this superquadric in its local (body) frame.

        For the world-frame mesh use :meth:`to_mesh`.

        Returns
        -------
        vertices : np.ndarray
            ``(resolution**2, 3)`` local-frame vertices.
        triangles : np.ndarray
            ``(2 * (resolution - 1)**2, 3)`` vertex-index triples.
        """
        eta = np.linspace(-np.pi / 2, np.pi / 2, resolution)
        omega = np.linspace(-np.pi, np.pi, resolution)
        eta, omega = np.meshgrid(eta, omega)

        x, y, z = _surface_xyz(self.scales, self.exponents, eta, omega)

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
                # Wind so face normals point outward (right-hand rule).
                triangles.append([v0, v2, v1])
                triangles.append([v1, v2, v3])

        return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)

    def generate_render_mesh(self, resolution=20):
        """Watertight local-frame triangle mesh with per-vertex analytic normals.

        Rendering-oriented variant of :meth:`generate_mesh`: the zero-area faces
        that the pole/seam snapping collapses are dropped (the surviving triangles
        close each pole with a fan, so the surface stays watertight), and every
        vertex carries the analytic outward surface normal — which stays
        well-defined at the poles, where a face normal computed from a collapsed
        triangle would be meaningless. Faces are wound counter-clockwise seen from
        outside, so renderers that cull back faces show the outside of the surface.

        Returns
        -------
        vertices : np.ndarray
            ``(resolution**2, 3)`` local-frame vertices.
        normals : np.ndarray
            ``(resolution**2, 3)`` unit outward normals, one per vertex.
        triangles : np.ndarray
            ``(n_faces, 3)`` vertex-index triples, ``n_faces <=
            2 * (resolution - 1)**2`` after the collapsed faces are removed.
        """
        eta = np.linspace(-np.pi / 2, np.pi / 2, resolution)
        omega = np.linspace(-np.pi, np.pi, resolution)
        eta, omega = np.meshgrid(eta, omega)

        x, y, z = _surface_xyz(self.scales, self.exponents, eta, omega)
        nx, ny, nz = _surface_normals(self.scales, self.exponents, eta, omega)
        vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
        normals = np.stack([nx.ravel(), ny.ravel(), nz.ravel()], axis=1)

        triangles = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                v0 = i * resolution + j
                v1 = i * resolution + (j + 1)
                v2 = (i + 1) * resolution + j
                v3 = (i + 1) * resolution + (j + 1)
                # Wind so face normals point outward (right-hand rule), like
                # generate_mesh, and skip the faces collapsed by the snapping.
                for tri in ([v0, v2, v1], [v1, v2, v3]):
                    a, b, c = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
                    if np.linalg.norm(np.cross(b - a, c - a)) < 1e-12:
                        continue
                    triangles.append(tri)

        return vertices, normals, np.asarray(triangles, dtype=np.int64)

    def to_mesh(self, resolution=20):
        """Triangle mesh of this superquadric in world coordinates.

        Returns
        -------
        vertices : np.ndarray
            ``(resolution**2, 3)`` array of world-frame vertices.
        triangles : np.ndarray
            ``(2 * (resolution - 1)**2, 3)`` array of vertex-index triples.
        """
        vertices, triangles = self.generate_mesh(resolution=resolution)
        world = self.transform_point_to_world(vertices.T).T
        return world, triangles

    def grad_inside_outside_wrt_point(self, point):
        """Gradient of the inside-outside function w.r.t. the world point.

        At axis-aligned points the normalized coordinates are clamped off the axis
        plane by ``_AXIS_EPS`` so the result stays finite (the singularity is
        removable for exponents <= 2; for exponents > 2 it is genuine and the
        clamped value is a large finite regularization).
        """
        local_point = self.transform_point_to_local(point)
        e1, e2 = self.exponents
        a, b, c = self.scales
        x, y, z = local_point

        # Clamp the normalized coordinates off the axis planes so the result stays
        # finite at axis-aligned points (see _clamp_off_axis / _AXIS_EPS).
        xs = _clamp_off_axis(x / a)
        ys = _clamp_off_axis(y / b)
        zs = _clamp_off_axis(z / c)

        f2D = (xs ** 2) ** (1 / e2) + (ys ** 2) ** (1 / e2)
        df2D_dx = 2 / e2 * (1 / a) * (xs ** 2) ** (1 / e2) / xs
        df2D_dy = 2 / e2 * (1 / b) * (ys ** 2) ** (1 / e2) / ys

        dfdx = (e2 / e1) * f2D ** (e2 / e1 - 1) * df2D_dx
        dfdy = (e2 / e1) * f2D ** (e2 / e1 - 1) * df2D_dy
        dfdz = (2 / e1) * (1 / c) * (zs ** 2) ** (1 / e1) / zs

        return np.array(self.rotation @ np.array([dfdx, dfdy, dfdz]))

    def hessian_inside_outside_wrt_point(self, point):
        """Hessian of the inside-outside function w.r.t. the world point.

        Axis-aligned points are clamped off the axis plane by ``_AXIS_EPS`` to keep
        the result finite, same regularization as the gradient.
        """
        local_point = self.transform_point_to_local(point)
        e1, e2 = self.exponents
        a, b, c = self.scales
        x, y, z = local_point

        # Clamp the normalized coordinates off the axis planes so the result stays
        # finite at axis-aligned points (see _clamp_off_axis / _AXIS_EPS).
        xs = _clamp_off_axis(x / a)
        ys = _clamp_off_axis(y / b)
        zs = _clamp_off_axis(z / c)

        f2D = (xs ** 2) ** (1 / e2) + (ys ** 2) ** (1 / e2)
        df2D_dx = 2 / e2 * (1 / a) * (xs ** 2) ** (1 / e2) / xs
        df2D_dy = 2 / e2 * (1 / b) * (ys ** 2) ** (1 / e2) / ys

        d2fdxdx = (
            e2 / e1 * (e2 / e1 - 1) * f2D ** (e2 / e1 - 2) * df2D_dx ** 2
            + (e2 / e1) * f2D ** (e2 / e1 - 1) * 2 / e2 * (1 / a ** 2)
            * (2 / e2 - 1) * (xs ** 2) ** (1 / e2) / xs ** 2
        )
        d2fdxdy = e2 / e1 * (e2 / e1 - 1) * f2D ** (e2 / e1 - 2) * df2D_dx * df2D_dy
        d2fdydy = (
            e2 / e1 * (e2 / e1 - 1) * f2D ** (e2 / e1 - 2) * df2D_dy ** 2
            + (e2 / e1) * f2D ** (e2 / e1 - 1) * 2 / e2 * (1 / b ** 2)
            * (2 / e2 - 1) * (ys ** 2) ** (1 / e2) / ys ** 2
        )
        d2fdzdz = 1 / c ** 2 * 2 / e1 * (2 / e1 - 1) * (zs ** 2) ** (1 / e1) / zs ** 2

        H_local = np.array([
            [d2fdxdx, d2fdxdy, 0.0],
            [d2fdxdy, d2fdydy, 0.0],
            [0.0, 0.0, d2fdzdz],
        ])
        return self.rotation @ H_local @ self.rotation.T
