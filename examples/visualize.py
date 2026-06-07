"""Render a row of superquadrics with both backends, matched side by side.

Interactive use is just::

    from superquadrics import Superquadric, SuperquadricShape, superquadric_plotter
    shape = SuperquadricShape(scales=[1, 1.5, 0.8], exponents=[0.6, 0.9])
    sq = Superquadric(shape, center=[0, 0, 0], rotation=[0.2, 0.4, -0.3])
    superquadric_plotter(sq, plotter="pyvista", opacity=0.5)   # or plotter="open3d"

This script additionally renders both backends off-screen to PNGs using a shared
orthographic camera (so the two images line up exactly) and trims whitespace.
Run::

    python examples/visualize.py --opacity 0.5 --out /tmp

Transparency note: PyVista uses depth peeling (true ordered transparency, so you
see through to the far surface), while Open3D (Filament) uses weighted-blended OIT
and culls back faces. This script renders Open3D back faces and brightens the alpha
so the two read similarly, but Open3D will not reproduce PyVista's crisp see-through.

Requires the ``[examples]`` extra (open3d + pyvista + pillow).
"""
import argparse
import multiprocessing as mp
import os

import numpy as np
from PIL import Image

from superquadrics import Superquadric, SuperquadricShape
from superquadrics.plotting import plot_quadric_open3d, plot_quadric_pyvista

# (label, scales, exponents, color) — the exponents pick the shape family.
SHAPES = [
    ("ellipsoid",    [0.6, 0.9, 0.5], [1.0, 1.0], (0.85, 0.20, 0.20)),
    ("rounded cube", [0.6, 0.6, 0.6], [0.3, 0.3], (0.20, 0.70, 0.30)),
    ("cylinder",     [0.5, 0.5, 0.9], [0.2, 1.0], (0.20, 0.45, 0.85)),
    ("pillow",       [0.7, 0.7, 0.4], [1.0, 0.4], (0.90, 0.65, 0.10)),
    ("pinched",      [0.7, 0.7, 0.7], [2.0, 2.0], (0.60, 0.30, 0.75)),
]
SPACING = 2.2
RESOLUTION = 90
SIZE = (1600, 900)          # (width, height) in pixels
DISTANCE = 20.0             # camera distance (orthographic, so only sign matters)

# Shared camera: same target / view direction / up for both backends.
TARGET = np.array([(len(SHAPES) - 1) * SPACING / 2.0, 0.0, 0.0])
FRONT = np.array([0.5, -0.7, 0.45]); FRONT = FRONT / np.linalg.norm(FRONT)
UP = np.array([0.0, 0.0, 1.0])


def build_scene():
    """Return a list of (Superquadric, color) placed in a tilted row."""
    scene = []
    for i, (_, scales, exps, color) in enumerate(SHAPES):
        euler = [0.3 * i, 0.2 * i, 0.15 * i]
        shape = SuperquadricShape(scales, exps)
        scene.append((Superquadric(shape, center=[i * SPACING, 0, 0], rotation=euler), color))
    return scene


def ortho_extent(scene, aspect, margin=1.05):
    """Half-height V of an orthographic frustum that fits the whole scene.

    Projects every mesh vertex onto the camera plane so both backends can use
    the identical world-units-per-pixel scale.
    """
    forward = -FRONT
    right = np.cross(forward, UP); right /= np.linalg.norm(right)
    true_up = np.cross(right, forward); true_up /= np.linalg.norm(true_up)
    half_w = half_h = 0.0
    for sq, _ in scene:
        verts, _ = sq.to_mesh(resolution=RESOLUTION)
        rel = verts - TARGET
        half_w = max(half_w, np.abs(rel @ right).max())
        half_h = max(half_h, np.abs(rel @ true_up).max())
    return max(half_h, half_w / aspect) * margin


def render_pyvista(scene, opacities, V):
    import pyvista as pv
    w, h = SIZE
    pv.OFF_SCREEN = True
    p = pv.Plotter(off_screen=True, window_size=(w, h))
    p.set_background("white")
    for (sq, color), opacity in zip(scene, opacities):
        verts, tris = sq.to_mesh(resolution=RESOLUTION)
        mesh = plot_quadric_pyvista(verts, tris, visualize=False)
        p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, specular=0.3)
    p.enable_depth_peeling()           # true ordered transparency (see-through)
    p.camera_position = [tuple(TARGET + FRONT * DISTANCE), tuple(TARGET), tuple(UP)]
    p.enable_parallel_projection()
    p.camera.parallel_scale = V        # exact orthographic scale
    return Image.fromarray(p.screenshot(return_img=True))


def render_open3d(scene, opacities, V, supersample=2):
    import open3d as o3d
    rendering = o3d.visualization.rendering
    w, h = SIZE
    aspect = w / h
    renderer = rendering.OffscreenRenderer(w * supersample, h * supersample)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])
    renderer.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, -FRONT)
    for i, ((sq, color), opacity) in enumerate(zip(scene, opacities)):
        # Open3D blends transparency more aggressively toward the background than
        # PyVista, so the same alpha looks much fainter. Compensate with sqrt()
        # purely so this side-by-side comparison reads consistently (the library
        # itself uses the literal opacity). opacity == 1 stays exactly opaque.
        o3d_opacity = opacity if opacity >= 1.0 else opacity ** 0.5
        verts, tris = sq.to_mesh(resolution=RESOLUTION)
        objects = plot_quadric_open3d(verts, tris, color=color, opacity=o3d_opacity,
                                      show_frames=False, visualize=False)
        for obj in objects:
            renderer.scene.add_geometry(f"{obj['name']}_{i}", obj["geometry"], obj["material"])
        # Open3D culls back faces, so a transparent mesh would otherwise show only
        # its front surface blended with the background. Add a winding-flipped copy
        # to draw the far surface too. NOTE: Open3D uses weighted-blended OIT, so
        # this still is not the true depth-peeled "see-through" PyVista produces.
        if o3d_opacity < 1.0:
            back = o3d.geometry.TriangleMesh()
            back.vertices = o3d.utility.Vector3dVector(verts)
            back.triangles = o3d.utility.Vector3iVector(np.asarray(tris)[:, ::-1])
            back.compute_vertex_normals()
            back.paint_uniform_color(list(color))
            renderer.scene.add_geometry(f"back_{i}", back, objects[0]["material"])
    cam = renderer.scene.camera
    cam.set_projection(rendering.Camera.Projection.Ortho,
                       -V * aspect, V * aspect, -V, V, 0.01, DISTANCE * 4)
    cam.look_at(TARGET, TARGET + FRONT * DISTANCE, UP)
    img = np.asarray(renderer.render_to_image())
    return Image.fromarray(img).resize(SIZE, Image.LANCZOS)   # supersample down -> AA


def crop_to_content(im, margin=20, bg=250):
    arr = np.asarray(im.convert("RGB"))
    mask = np.any(arr < bg, axis=2)
    if not mask.any():
        return im
    ys, xs = np.where(mask)
    x0 = max(0, xs.min() - margin)
    y0 = max(0, ys.min() - margin)
    x1 = min(arr.shape[1], xs.max() + 1 + margin)
    y1 = min(arr.shape[0], ys.max() + 1 + margin)
    return im.crop((x0, y0, x1, y1))


def pad_to(im, size, bg=(255, 255, 255)):
    canvas = Image.new("RGB", size, bg)
    canvas.paste(im, ((size[0] - im.width) // 2, (size[1] - im.height) // 2))
    return canvas


def _render_backend(backend, min_opacity, raw_path):
    """Render one backend and save a raw PNG. Run in its own process: PyVista
    (VTK/OpenGL) and Open3D (Filament/EGL) cannot share a process safely."""
    scene = build_scene()
    V = ortho_extent(scene, aspect=SIZE[0] / SIZE[1])
    # Transparency decreases left -> right: leftmost = min_opacity, rightmost opaque.
    opacities = np.linspace(min_opacity, 1.0, len(scene))
    render = render_pyvista if backend == "pyvista" else render_open3d
    render(scene, opacities, V).save(raw_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opacity", type=float, default=0.3,
                        help="opacity of the leftmost shape (0-1); ramps up to 1.0 "
                             "on the right, so transparency decreases left to right")
    parser.add_argument("--out", default=".", help="output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Render each backend in a fresh process to keep their GL contexts apart.
    ctx = mp.get_context("spawn")
    raws = {}
    for backend in ("pyvista", "open3d"):
        raw = os.path.join(args.out, f"_raw_{backend}.png")
        proc = ctx.Process(target=_render_backend, args=(backend, args.opacity, raw))
        proc.start()
        proc.join()
        if proc.exitcode == 0 and os.path.exists(raw):
            raws[backend] = raw
        else:
            print(f"warning: {backend} render failed (exit code {proc.exitcode})")

    images = {name: crop_to_content(Image.open(raw)) for name, raw in raws.items()}
    if not images:
        return
    common = (max(im.width for im in images.values()),
              max(im.height for im in images.values()))
    for name, im in images.items():
        path = os.path.join(args.out, f"superquadrics_{name}.png")
        pad_to(im, common).save(path)
        os.remove(raws[name])
        print(f"{name} -> {path}")


if __name__ == "__main__":
    main()
