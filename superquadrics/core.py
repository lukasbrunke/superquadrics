"""Superquadric geometry: the canonical Superquadric class and mesh helpers.

Pure geometry only — depends on numpy and scipy, never on visualization libraries.
Convention: a superquadric is described by ``center`` (world position), ``scales``
``[a, b, c]``, shape ``exponents`` ``[e1, e2]``, and a ``rotation`` (3x3 matrix,
Euler 'xyz' angles, or [x, y, z, w] quaternion).
"""

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


def _clamp_off_axis(t):
    """Sign-preserving clamp of a normalized coordinate away from 0 by ``_AXIS_EPS``."""
    if abs(t) < _AXIS_EPS:
        return _AXIS_EPS if t >= 0 else -_AXIS_EPS
    return t


class Superquadric:
    """A superquadric with arbitrary position and orientation."""

    def __init__(self, center, scales, exponents, rotation=None):
        """
        Args:
            center: [x, y, z] world coordinates of the centre.
            scales: [a, b, c] scale parameters.
            exponents: [e1, e2] shape parameters.
            rotation: one of a 3x3 matrix, Euler 'xyz' angles (len 3),
                or an [x, y, z, w] quaternion (len 4). Defaults to identity.
        """
        self.center = np.array(center, dtype=float)
        self.scales = np.array(scales, dtype=float)
        self.exponents = np.array(exponents, dtype=float)

        if rotation is None:
            self.rotation = np.eye(3)
        elif isinstance(rotation, (list, np.ndarray)):
            rotation = np.array(rotation, dtype=float)
            if rotation.shape == (3, 3):
                self.rotation = rotation
            elif rotation.shape == (3,):
                self.rotation = Rotation.from_euler("xyz", rotation).as_matrix()
            elif rotation.shape == (4,):
                self.rotation = Rotation.from_quat(rotation).as_matrix()
            else:
                raise ValueError("Invalid rotation format")
        else:
            raise ValueError("Invalid rotation format")

    @property
    def pose(self):
        """4x4 homogeneous transform mapping the local (body) frame to the world."""
        T = np.eye(4)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.center
        return T

    @property
    def pose_inverse(self):
        """4x4 homogeneous transform mapping the world frame to the local frame."""
        T = np.eye(4)
        T[:3, :3] = self.rotation.T
        T[:3, 3] = -self.rotation.T @ self.center
        return T

    def transform_point_to_local(self, point):
        """World -> local. Accepts a single (3,) point or a (3, N) array."""
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

    def get_surface_points(self, n_points=20, scaling=None, n_v=None, n_u=None, mode="simple"):
        """Return (x, y, z) grids of world-frame surface points, each shaped (n_v, n_u).

        ``n_u`` samples the u parameter in [-pi/2, pi/2]; ``n_v`` samples v in
        [-pi, pi]; both default to ``n_points``. mode='simple' uses uniform angle
        sampling; mode='uniform' reparameterizes by arc length for more even
        spacing. ``scaling`` (if not None) inflates the surface by
        ``scaling ** (e1/2)`` (used for the HOCBF scaled-superquadric construction).
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
            x = a1 * sign_pow(np.cos(u), e1) * sign_pow(np.cos(v), e2)
            y = a2 * sign_pow(np.cos(u), e1) * sign_pow(np.sin(v), e2)
            z = a3 * sign_pow(np.sin(u), e1)
        elif mode == "uniform":
            u_fine = np.linspace(-np.pi / 2, np.pi / 2, 1000)
            v_fine = np.linspace(-np.pi, np.pi, 1000)

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
            x = a1 * sign_pow(np.cos(u), e1) * sign_pow(np.cos(v), e2)
            y = a2 * sign_pow(np.cos(u), e1) * sign_pow(np.sin(v), e2)
            z = a3 * sign_pow(np.sin(u), e1)
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
        a1, a2, a3 = self.scales
        e1, e2 = self.exponents

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

        return np.asarray(vertices, dtype=float), np.asarray(triangles, dtype=np.int64)

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
