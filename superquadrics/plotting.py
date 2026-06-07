"""Open3D and PyVista visualization for superquadrics.

Visualization libraries are imported lazily so the core package can be used
without installing the optional ``[viz]`` dependencies. Both backends support
transparency via an ``opacity`` argument (Open3D uses its modern rendering API).
"""

import numpy as np

from .core import Superquadric


def plot_quadric_open3d(vertices, triangles, color=(1.0, 0.0, 0.0), opacity=1.0,
                        frame_size=1.0, show_frames=True, visualize=True):
    """Build (and optionally display) an Open3D mesh via the modern rendering API.

    Returns a list of ``{"name", "geometry", "material"}`` dicts, the format
    accepted by ``o3d.visualization.draw`` and ``rendering.OffscreenRenderer``.
    ``opacity`` < 1 uses the transparency shader (the legacy ``draw_geometries``
    path could not render transparency).
    """
    import open3d as o3d
    rendering = o3d.visualization.rendering

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(list(color))

    material = rendering.MaterialRecord()
    material.shader = "defaultLitTransparency" if opacity < 1.0 else "defaultLit"
    r, g, b = color
    material.base_color = [r, g, b, float(opacity)]

    objects = [{"name": "superquadric", "geometry": mesh, "material": material}]
    if show_frames:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
        frame_material = rendering.MaterialRecord()
        frame_material.shader = "defaultLit"
        objects.append({"name": "frame", "geometry": frame, "material": frame_material})

    if visualize:
        o3d.visualization.draw(objects)

    return objects


def plot_quadric_pyvista(vertices, triangles, color=(1.0, 0.0, 0.0),
                         opacity=1.0, visualize=True):
    """Build (and optionally show) a PyVista mesh from explicit triangles.

    Uses the superquadric triangulation directly rather than a fragile
    ``delaunay_3d`` reconstruction. ``opacity`` < 1 renders the mesh transparent.
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
    """Plot a Superquadric with the chosen backend ('open3d' or 'pyvista').

    Both backends honour ``opacity`` (0-1) for transparency.
    """
    vertices, triangles = sq.to_mesh(resolution=resolution)
    if plotter == "open3d":
        return plot_quadric_open3d(vertices, triangles, color=color,
                                   opacity=opacity, visualize=visualize)
    elif plotter == "pyvista":
        return plot_quadric_pyvista(vertices, triangles, color=color,
                                    opacity=opacity, visualize=visualize)
    raise ValueError(f"Invalid plotter: {plotter!r}")
