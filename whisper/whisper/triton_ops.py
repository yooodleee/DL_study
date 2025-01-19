from functools import lru_cache

import numpy as np
import torch

try:
    import triton
    import trition.language as tl
except ImportError:
    raise RuntimeError(
        "triton import failed; try `pip install --pre triton`")

