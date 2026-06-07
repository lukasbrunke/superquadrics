"""Open3D and PyVista visualization for superquadrics.

Visualization libraries are imported lazily so the core package can be used
without installing the optional ``[viz]`` dependencies.
"""

import warnings

import numpy as np

from .core import Superquadric, generate_superquadric_mesh


def superquadric_to_mesh(sq: Superquadric, resolution: int = 20):
    """Return (vertices_world, triangles) for ``sq`` as numpy arrays.

    Builds the canonical local mesh via ``generate_superquadric_mesh`` and
    transforms the vertices into world coordinates.
    """
    a1, a2, a3 = sq.scales
    e1, e2 = sq.exponents
    vertices, triangles = generate_superquadric_mesh(a1, a2, a3, e1, e2, resolution=resolution)
    vertices = np.asarray(vertices, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    world = sq.transform_point_to_world(vertices.T).T
    return world, triangles


def plot_quadric_open3d(vertices, triangles, color=(1.0, 0.0, 0.0),
                        frame_size=1.0, show_frames=True, visualize=True):
    """Render/return an Open3D triangle mesh for the given vertices/triangles."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))

    geometries = [mesh]
    if show_frames:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        geometries.append(frame)

    if visualize:
        o3d.visualization.draw_geometries(geometries)

    return geometries


def plot_quadric_pyvista(vertices, triangles, color=(1.0, 0.0, 0.0),
                         opacity=1.0, visualize=True):
    """Build (and optionally show) a PyVista mesh from explicit triangles.

    Uses the superquadric triangulation directly rather than a fragile
    ``delaunay_3d`` reconstruction.
    """
    import pyvista as pv

    triangles = np.asarray(triangles, dtype=np.int64)
    faces = np.hstack([np.full((len(triangles), 1), 3, dtype=np.int64), triangles]).ravel()
    mesh = pv.PolyData(np.asarray(vertices, dtype=float), faces)

    if visualize:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, opacity=opacity, color=color)
        plotter.show()

    return mesh


def superquadric_plotter(sq: Superquadric, plotter="pyvista", resolution=20,
                         color=(1.0, 0.0, 0.0), opacity=1.0, visualize=True):
    """Plot a Superquadric with the chosen backend ('open3d' or 'pyvista')."""
    vertices, triangles = superquadric_to_mesh(sq, resolution=resolution)
    if plotter == "open3d":
        if opacity < 1.0:
            warnings.warn("Open3D mesh rendering does not support opacity; ignoring it.")
        return plot_quadric_open3d(vertices, triangles, color=color, visualize=visualize)
    elif plotter == "pyvista":
        return plot_quadric_pyvista(vertices, triangles, color=color,
                                    opacity=opacity, visualize=visualize)
    raise ValueError(f"Invalid plotter: {plotter!r}")
