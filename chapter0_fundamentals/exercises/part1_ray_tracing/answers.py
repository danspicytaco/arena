# %%
import sys
from pathlib import Path

import torch as t
from jaxtyping import Float
from torch import Tensor

# This works whether VS Code runs a cell from the repository root or this exercise directory.
chapter = "chapter0_fundamentals"
root_dir = next(path for path in Path(__file__).resolve().parents if (path / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import render_lines_with_plotly


# %%
# At x=1, rays should cover y coordinates from -y_limit to +y_limit, inclusively.
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.
    """
    # Create the output Tensor.
    rays = t.zeros((num_pixels, 2, 3), dtype=t.float32)

    # Evenly space num_pixels points with linspace.
    y_direction_points = t.linspace(-y_limit, y_limit, num_pixels)

    # Fill the direction array x-axis and y-axis with coordinates.
    rays[:, 1, 0] = 1
    rays[:, 1, 1] = y_direction_points

    # To save space complexity, this can instead be:
    # t.linspace(-y_limit, y_limit, num_pixels, out=rays[:, 1, 1])
    return rays


# %%
rays1d = make_rays_1d(9, 10.0)
render_lines_with_plotly(rays1d)


# %%
# Solve O + uD = L_1 + v(L_2 - L_1). An intersection requires u >= 0 and 0 <= v <= 1.
def intersect_ray_1d(
    ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]
) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O = ray[0]
    D = ray[1]
    L_1 = segment[0]
    L_2 = segment[1]

    # Create the tensors according to the formula above as AX = B.
    A = t.tensor([[D[0], L_1[0] - L_2[0]], [D[1], L_1[1] - L_2[1]]])
    B = t.tensor([[L_1[0] - O[0]], [L_1[1] - O[1]]])

    # Solve according to AX = B.
    X = t.linalg.solve(A, B)

    # If X = [[u], [v]], check whether it lies in the ray and segment ranges.
    # u = X[0, 0]; v = X[1, 0]
    intersects = X[0, 0] >= 1 and X[1, 0] >= 0 and X[1, 0] <= 1
    return intersects


# %%
tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)
