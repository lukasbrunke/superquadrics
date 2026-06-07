"""Superquadric geometry and visualization."""

from .core import Superquadric, SuperquadricShape, sign_pow
from .plotting import (
    plot_quadric_open3d,
    plot_quadric_pyvista,
    superquadric_plotter,
)

__all__ = [
    "Superquadric",
    "SuperquadricShape",
    "sign_pow",
    "plot_quadric_open3d",
    "plot_quadric_pyvista",
    "superquadric_plotter",
]
