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
