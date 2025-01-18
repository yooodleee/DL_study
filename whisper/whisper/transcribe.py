import argparse
import os
import traceback
import warnings
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import torch
import tqdm

from .audio import (
    FRAMES_PER_SECOND,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    log_mel_spectrogram,
    pad_or_trim)

from .decoding import DecodingOptions, DecodingResult
from .timing import add_word_timestapms
from .tokenizer import LANGUAGES, TO_LANGUAGE_CODE, get_tokenizer
from .utils import (
    exact_div,
    format_timestamp,
    get_end,
    get_writer,
    make_safe,
    optional_float,
    optional_int,
    str2bool)

if TYPE_CHECKING:
    from .model import Whisper


def transcribe(
        model: "Whisper",
        audio: Union[str, np.ndarray, torch.Tensor],
        *,
        verbose: Optional[bool] = None,
        temperature: Union[float, Tuple[float, ...]] = \
                        (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        compression_ratio_threashold: Optional[float] = 2.4,
        logprob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        initial_prompt: Optional[str] = None,
        carry_initial_prompt: bool = False,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        clip_timestamps: Union[str, List[float]] = 0,
        hallucination_silence_threshold: Optional[float] = None,
        **decode_options):
    """
    Transcribe an audion file using Whisper

    Parameters:
        model: Whisper
            The Whisper model instance
        audio: Union[str, np.ndarray, torch.Tensor]
            The path to the audio file to open, or the audio waveform
        verbose: bool
            Whether to display the text being decoded to the console. If True,
                displays all the details, If False, displays minimal details.
                If None, does not display anything
        
    """