import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .utils import exact_div

# hard-coaded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
# 480000 samples in a 30-second chunk
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
# 3000 frames in a mel spectrogram input
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

# the initial convolutions has stride 2
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2
# 10ms per audio frame
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)
# 20ms per audio token
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)


