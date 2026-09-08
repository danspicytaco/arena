# %%
import io
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import httpx
import pandas as pd
import tqdm
from dotenv import load_dotenv
from IPython.display import HTML, display
from openai import OpenAI

# Make sure exercises are in the path
chapter = "chapter4_alignment_science"
section = "part2_science_of_misalignment"
root_dir = next(p for p in Path(__file__).resolve().parents if p.name == "ARENA_3.0")
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

from part2_science_of_misalignment import tests, utils


# %%
