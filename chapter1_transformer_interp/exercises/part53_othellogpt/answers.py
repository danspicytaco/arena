# %%
import copy
import os
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import numpy as np
import pandas as pd
import plotly.express as px
import torch as t
import wandb
from eindex import eindex
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm.notebook import tqdm
from transformer_lens import ActivationCache, HookedTransformer, HookedTransformerConfig
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import download_file_from_hf, get_act_name, to_numpy

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter1_transformer_interp"
section = "part53_othellogpt"
root_dir = next(p for p in Path(__file__).resolve().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part53_othellogpt.tests as tests
import part53_othellogpt.utils as utils
from neel_plotly import scatter

t.set_grad_enabled(False)


# %%
