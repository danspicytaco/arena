# %%
import math
import os
import sys
from pathlib import Path

import einops
import numpy as np
import torch as t
from torch import Tensor

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part0_prereqs"
root_dir = next(p for p in Path(__file__).resolve().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part0_prereqs.tests as tests
from part0_prereqs.utils import display_array_as_img, display_soln_array_as_img


# %%
# einops operations

arr = np.load(section_dir / "numbers.npy")

arr1 = einops.rearrange(arr, "b c h w -> c (b h) w")
display_array_as_img(arr1)

arr2 = einops.repeat(arr[0], "c h w -> c (2 h) w")
display_array_as_img(arr2)


# %%
# einops operations & broadcasting


def assert_all_equal(actual: Tensor, expected: Tensor) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    assert (actual == expected).all(), f"Value mismatch, got: {actual}"
    print("Tests passed!")


def assert_all_close(actual: Tensor, expected: Tensor, atol=1e-3) -> None:
    assert actual.shape == expected.shape, f"Shape mismatch, got: {actual.shape}"
    t.testing.assert_close(actual, expected, atol=atol, rtol=0.0)
    print("Tests passed!")


# %%
# A1 rearrange


def rearrange_1() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[3, 4],
     [5, 6],
     [7, 8]]
    """
    initial_tensor = t.arange(3, 9)
    result = einops.rearrange(initial_tensor, "(v1 v2) -> v1 v2", v1=3, v2=2)
    return result


expected = t.tensor([[3, 4], [5, 6], [7, 8]])
assert_all_equal(rearrange_1(), expected)


# %%
# A2 rearrange
def rearrange_2() -> Tensor:
    """Return the following tensor using only t.arange and einops.rearrange:

    [[1, 2, 3],
     [4, 5, 6]]
    """
    initial_tensor = t.arange(1, 7)
    result = einops.rearrange(initial_tensor, "(v1 v2) -> v1 v2", v1=2, v2=3)
    return result


assert_all_equal(rearrange_2(), t.tensor([[1, 2, 3], [4, 5, 6]]))


# %%
# B1 temperature average
def temperatures_average(temps: Tensor) -> Tensor:
    """Return the average temperature for each week.

    temps: a 1D temperature containing temperatures for each day.
    Length will be a multiple of 7 and the first 7 days are for the first week, second 7 days for the second week, etc.

    You can do this with a single call to reduce.
    """
    assert len(temps) % 7 == 0
    return einops.reduce(temps, "(week day) -> week", "mean", day=7)


temps = t.tensor([71, 72, 70, 75, 71, 72, 70, 75, 80, 85, 80, 78, 72, 83]).float()
expected = [71.571, 79.0]
assert_all_close(temperatures_average(temps), t.tensor(expected))


# %%
# B2 temperature difference
def temperatures_differences(temps: Tensor) -> Tensor:
    """For each day, subtract the average for the week the day belongs to.

    temps: as above
    """
    assert len(temps) % 7 == 0
    avg_temps = temperatures_average(temps)  # (2,)
    rep_avg_temps = einops.repeat(avg_temps, "week -> (week day)", day=7)
    return temps - rep_avg_temps


expected = [
    -0.571,
    0.429,
    -1.571,
    3.429,
    -0.571,
    0.429,
    -1.571,
    -4.0,
    1.0,
    6.0,
    1.0,
    -1.0,
    -7.0,
    4.0,
]
actual = temperatures_differences(temps)
assert_all_close(actual, t.tensor(expected))

# %%
# B3 temperature normalized


def temperatures_normalized(temps: Tensor) -> Tensor:
    """For each day, subtract the weekly average and divide by the weekly standard deviation.

    temps: as above

    Pass t.std to reduce.
    """
    diffs = temperatures_differences(temps)
    weekly_std_dev = einops.reduce(temps, "(week day) -> week", t.std, day=7)
    weekly_std_dev_rep = einops.repeat(weekly_std_dev, "week -> (week day)", day=7)
    return diffs / weekly_std_dev_rep


expected = [
    -0.333,
    0.249,
    -0.915,
    1.995,
    -0.333,
    0.249,
    -0.915,
    -0.894,
    0.224,
    1.342,
    0.224,
    -0.224,
    -1.565,
    0.894,
]
actual = temperatures_normalized(temps)
assert_all_close(actual, t.tensor(expected))

# %%
# C1 normalize a matrix


def normalize_rows(matrix: Tensor) -> Tensor:
    """Normalize each row of the given 2D matrix.

    matrix: a 2D tensor of shape (m, n).

    Returns: a tensor of the same shape where each row is divided by its l2 norm.
    """
    raise NotImplementedError()


matrix = t.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).float()
expected = t.tensor([[0.267, 0.535, 0.802], [0.456, 0.570, 0.684], [0.503, 0.574, 0.646]])
assert_all_close(normalize_rows(matrix), expected)

# %%
