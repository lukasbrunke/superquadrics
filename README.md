# superquadrics

Superquadric geometry and Open3D / PyVista visualization.

## Install

```bash
pip install -e ".[viz]"     # with plotting backends
pip install -e .            # geometry only (numpy + scipy)
```

## Usage

```python
import numpy as np
from superquadrics import Superquadric, superquadric_plotter

sq = Superquadric(center=[0, 0, 0], scales=[1.0, 1.5, 0.8], exponents=[0.6, 0.9],
                  rotation=[0.0, 0.0, 0.0])   # matrix, Euler 'xyz', or [x,y,z,w] quat

print(sq.inside_outside_function(np.array([0.5, 0.0, 0.0])))  # <1 inside, >1 outside
g = sq.grad_inside_outside_wrt_point(np.array([0.7, -0.9, 1.3]))
H = sq.hessian_inside_outside_wrt_point(np.array([0.7, -0.9, 1.3]))

superquadric_plotter(sq, plotter="pyvista")   # or plotter="open3d"
```

## Scope

This package is geometry + plotting only. Distance/scaling, CBF safety filters,
robot kinematics, and ROS integration live in separate repositories.
