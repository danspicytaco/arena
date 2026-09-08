# %%
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from IPython.display import display
from openai import OpenAI
from scipy import stats

# Make sure exercises are in the path
chapter = "chapter4_alignment_science"
section = "part5_investigator_agents"
root_dir = next(p for p in Path(__file__).resolve().parents if p.name == "ARENA_3.0")
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section

if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))

import part5_investigator_agents.tests as tests
import part5_investigator_agents.utils as utils


# %%
