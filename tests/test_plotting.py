import numpy as np
import pytest

from superquadrics.core import Superquadric
from superquadrics.plotting import (
    superquadric_to_mesh,
    plot_quadric_open3d,
    plot_quadric_pyvista,
    superquadric_plotter,
)


def _sq():
    return Superquadric([0.0, 0.0, 0.0], [1.0, 1.5, 0.8], [0.6, 0.9])


def test_superquadric_to_mesh_returns_world_mesh():
    sq = _sq()
    vertices, triangles = superquadric_to_mesh(sq, resolution=8)
    assert vertices.shape == (64, 3)
    assert triangles.shape == (2 * 49, 3)


def test_plot_pyvista_builds_polydata_without_gui():
    pv = pytest.importorskip("pyvista")
    vertices, triangles = superquadric_to_mesh(_sq(), resolution=8)
    mesh = plot_quadric_pyvista(vertices, triangles, visualize=False)
    assert mesh.n_points == 64
    assert mesh.n_cells == 2 * 49


def test_plot_open3d_builds_geometry_without_gui():
    pytest.importorskip("open3d")
    vertices, triangles = superquadric_to_mesh(_sq(), resolution=8)
    geometries = plot_quadric_open3d(vertices, triangles, visualize=False)
    assert len(geometries) >= 1


def test_superquadric_plotter_dispatch_without_gui():
    pytest.importorskip("pyvista")
    mesh = superquadric_plotter(_sq(), plotter="pyvista", resolution=8, visualize=False)
    assert mesh.n_points == 64


def test_superquadric_plotter_rejects_unknown_backend():
    with pytest.raises(ValueError):
        superquadric_plotter(_sq(), plotter="matplotlib", visualize=False)
