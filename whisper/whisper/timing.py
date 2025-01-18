import itertools
import subprocess
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List

import numba
import numpy as np
import torch
import torch.nn.functional as F

from .audio import HOP_LENGTH, SAMPLE_RATE, TOKENS_PER_SECOND
from .tokenizer import Tokenizer

if TYPE_CHECKING:
    from .model import Whisper


def median_filter(x: torch.Tensor, filter_width: int):
    """
    Apply a median filter of width `filter_width` along the last dimension
        of `x`
    """
    pad_width = filter_width // 2
    if x.shape[-1] <= pad_width:
        # F.pad requires the padding with to be smaller than the 
        # input dimension
        return x
    
    if (ndim := x.ndim) <= 2:
        # `F.pad` does not support 1D or 2D inputs for reflext padding
        # but supports 3D and 4D
        x = x[None, None, :]
    
    assert (filter_width > 0 and filter_width % 2 == 1), \
    "`filter_width` should be an odd number"

    result = None
    x = F.pad(x, (filter_width // 2, filter_width // 2, 0, 0), mode="reflect")
    if x.is_cuda:
        try:
            from .train_ops import media_filter_cuda

            result = media_filter_cuda(x, filter_width)
        except (RuntimeError, subprocess.CalledProcessError):
            warnings.warn(
                "Failed to launch Triton kernels, likely due to missing CUDA "
                "toolkit; falling back to a slower median kernel "
                "implementation...")
    
    if result is None:
        # sort() is faster than torch.median
        # (https://github.com/pytorch/pytorch/issues/51450)
        result = x.unfold(-1, 
                          filter_width, 1).sort()[0][..., filter_width // 2]
    
    if ndim <= 2:
        result = result[0, 0]
    
    return result


@numba.jit(nonpython=True)
def backtrace(trace: np.ndarray):
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1

    result = []
    while i > 0 or j > 0:
        result.append((i - 1, j - 1))

        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise ValueError("Unexpected trace[i, j]")
    
    result = np.array(result)
    return result[::-1, :].T


