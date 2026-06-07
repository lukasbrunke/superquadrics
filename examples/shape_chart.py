"""Standard superquadric shape chart (Barr / EMS figure): an e1 x e2 grid.

Each cell is a unit superquadric rendered with PyVista (semi-transparent, depth
-peeled, so the 3D structure is visible); the labeled grid is laid out with
matplotlib. e1 (rows) and e2 (columns) step from 0 to 2 by 0.5, with the origin
(0, 0) in the bottom-left. Exponents must be > 0, so the nominal 0 is rendered as
a small epsilon -- giving the cube corner, the sphere at (1, 1), and the
octahedron at (2, 2).

    pip install -e ".[examples]"
    python examples/shape_chart.py --out examples/images
"""
import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")            # CPU backend; no GL conflict with PyVista
import matplotlib.pyplot as plt

from superquadrics import Superquadric, SuperquadricShape
from superquadrics.plotting import plot_quadric_pyvista

LABELS = [0.0, 0.5, 1.0, 1.5, 2.0]   # axis tick labels (0 in bottom-left)
EPS = 0.01                           # nominal 0 rendered as this (exponents > 0)
RES = 100                            # mesh resolution (high -> crisp edges)
OPACITY = 0.6
COLOR = (0.30, 0.55, 0.78)
CELL_PX = 360
PARALLEL_SCALE = 1.6                 # fixed ortho scale -> all shapes same size
ROTATION_DEG = (0.0, 0.0, 30.0)      # Euler 'xyz' rotation applied to every shape


def render_cell(e1, e2):
    """Render one unit superquadric to an RGB image (white bg, transparent shape)."""
    import pyvista as pv
    pv.OFF_SCREEN = True
    rotation = [math.radians(a) for a in ROTATION_DEG]
    sq = Superquadric(SuperquadricShape([1, 1, 1], [e1, e2]), rotation=rotation)
    verts, tris = sq.to_mesh(resolution=RES)
    mesh = plot_quadric_pyvista(verts, tris, visualize=False)
    pl = pv.Plotter(off_screen=True, window_size=(CELL_PX, CELL_PX))
    pl.set_background("white")
    pl.enable_depth_peeling()        # true ordered transparency (see-through)
    pl.add_mesh(mesh, color=COLOR, opacity=OPACITY, smooth_shading=False, specular=0.2)
    pl.enable_parallel_projection()
    pl.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]   # consistent iso view
    pl.camera.parallel_scale = PARALLEL_SCALE
    img = pl.screenshot(return_img=True)
    pl.close()
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=".", help="output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    n = len(LABELS)
    cells = {(e1, e2): render_cell(max(e1, EPS), max(e2, EPS))
             for e1 in LABELS for e2 in LABELS}

    fig, axes = plt.subplots(n, n, figsize=(2.0 * n, 2.0 * n))
    for r in range(n):                  # image row 0 = top
        e1 = LABELS[n - 1 - r]          # bottom row -> e1 = 0
        for c in range(n):
            e2 = LABELS[c]              # left column -> e2 = 0
            ax = axes[r][c]
            ax.imshow(cells[(e1, e2)])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == n - 1:              # bottom row: e2 tick labels
                ax.set_xlabel(f"{e2:g}", fontsize=13)
            if c == 0:                  # left column: e1 tick labels
                ax.set_ylabel(f"{e1:g}", fontsize=13, rotation=0,
                              labelpad=12, va="center")

    fig.subplots_adjust(left=0.11, right=0.99, top=0.99, bottom=0.10,
                        wspace=0.0, hspace=0.0)
    fig.supxlabel(r"$\epsilon_2$", fontsize=18, y=0.02)    # further down
    fig.supylabel(r"$\epsilon_1$", fontsize=18, x=0.02)    # further left

    path = os.path.join(args.out, "shape_chart.png")
    fig.savefig(path, dpi=130)
    print("shape chart ->", path)


if __name__ == "__main__":
    main()
