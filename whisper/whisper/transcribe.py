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
    SAMPLE_RATE,
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

        temperature: Union[float, Tuple[float, ...]]
            Temperature for sampling. It can be a tuple of temperatures, which
                will be successively used upon failures according to either
                `compression_ratio_threshold` or `logprob_threshold`.

        compression_ratio_threshold: float
            If the gzip compression ratio is above this value, treat as failed

        logprob_threshold: float
            If the average log probability over sampled tokens is below this
                value, treat as failed
        
        no_speech_threshold: float
            If the no_speech probability is higher than this value AND the 
                average log probability over sampled tokens is below 
                `logprob_threshold`, consider the segment as silent

        condition_on_previous_text: bool
            if True, the previous output of the model is provided as a prompt
                for the next window; disabling may make the text inconsistent
                across windows, but the model becomes less prone to getting
                stuck in a failure loop, such as repetition looping or
                timestamps going out of sync.

        word_timestamps: bool
            Extract word-level timestamps using the cross-attention pattern
                and dynamic time warping, and include the timestamps for each
                word in each segment.

        prepend_punctuations: str
            If word_timestamps is True, merge these punctuation symbols with 
                the next word

        append_punctuations: str
            If word_timestamps is True,, merge these punctuation symbols with
                the previous word

        initial_prompt: Optional[str]
            Optional text to provide as a prompt for the first window. 
                This can be used to provide, or "prompt-engineer" a context
                transcription, e.g. custom vocabularies or proper nouns to 
                make it more likely to predict those word correctly.

        caarry_initial_prompt: bool
            If carry_initial_prompt is True, `initial_prompt` is prepended to 
                the prompt of each internal `decode()` call. If there is not
                enough context space at the start of the prompt, it is 
                left-sliced to make space.
        
        decode_options: dict
            Keyword arguments to construct `DecodingOptions` instances

        clip_timestamps: Union[str, List[float]]
            Comma-separated list start, end, start, end,... timestamps 
                (in seconds) of clips to process. The last end timestamp
                defaults to the end of the file.

        hallucination_silence_threshold: Optional[float]
            When word_timestamps is True, skip silent periods longer than 
                this threshold (in seconds) when a possible hallucination is
                detected

    Returns:
        A dictionary containing the resulting text ("text") and segment-level
            details ("segments"), and the spoken language ("language"), which 
            is detected when `decode_options["language"]` is None.
    """
    dtype = torch.float16 \
            if decode_options.get("fp16", True) \
            else torch.float32
    if model.device == torch.device("cpu"):
        if torch.cuda.is_available():
            warnings.warn(
                "Performing inference on CPU when CUDA is available")
        if dtype == torch.float16:
            warnings.warn(
                "FP16 is not supported on CPU; using FP32 instead")
            dtype = torch.float32
    
    if dtype == torch.float32:
        decode_options["fp16"] = False
    
    # Pad 30-seconds of silence to the input audio, for slicing
    mel = log_mel_spectrogram(audio,
                              model.dims.n_mels,
                              padding=N_SAMPLES)
    content_frames = mel.shape[-1] - N_SAMPLES
    content_durations = float(content_frames * HOP_LENGTH / SAMPLE_RATE)

    if decode_options.get("language", None) is None:
        if not model.is_multilingual:
            decode_options["language"] = "en"
        else:
            if verbose:
                print(
                    "Detecting language using up the first 30 seconds. "
                    "Use `--language` to specify the language")
            
            mel_segment = pad_or_trim(mel, 
                                      N_FRAMES).to(model.device).to(dtype)
            _, probs = model.detect_language(mel_segment)
            decode_options["language"] = max(probs, key=probs.get)
            if verbose is not None:
                print(
                    f"Detected language: {LANGUAGES[decode_options
                                                    ['language']].title()}")
    
    language: str = decode_options["language"]
    task: str = decode_options.get("task", "transcribe")
    tokensizer = get_tokenizer(
                    model.is_multilingual,
                    num_languages=model.num_languages,
                    language=language,
                    task=task)
    
    if isinstance(clip_timestamps, str):
        clip_timestamps = [float(ts) for ts in (clip_timestamps.split(",")
                                                if clip_timestamps else [])]
    
    seek_points: List[int] = [round(ts * FRAMES_PER_SECOND)
                              for ts in clip_timestamps]
    if len(seek_points) == 0:
        seek_points.append(0)
    if len(seek_points) % 2 == 1:
        seek_points.append(content_frames)
    seek_clips: List[Tuple[int, int]] = list(zip(seek_points[::2],
                                                 seek_points[1::2]))
    
    punctuation = "\"'“¿([{-\"'.。,，!！?？:：”)]}、"

    if word_timestamps and task == "transcribe":
        warnings.warn(
            "Word-level timestamps on translations may not be reliable.")
    
    def decode_with_fallback(segment: torch.Tensor) -> DecodingResult:
        temperatures = ([temperature]
                        if isinstance(temperature, (int, float))
                        else temperature)
        decode_result = None

        for t in temperatures:
            kwars = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwars.pop("beam_size", None)
                kwars.pop("patience", None)
            else:
                # disable best_of when t == -
                kwars.pop("best_of", None)
            
            options = DecodingOptions(**kwars, temperature=t)
            decode_result = model.decode(segment, options)

            needs_fallback = False
            if (compression_ratio_threashold is not None
                and decode_result.compression_ratio
                > compression_ratio_threashold):
                needs_fallback = True   # too repetitive
            if (logprob_threshold is not None
                and decode_result.avg_logprob
                < logprob_threshold):
                needs_fallback = True   # average log probability is too low
            if (no_speech_threshold is not None
                and decode_result.no_speech_prob
                > no_speech_threshold
                and logprob_threshold is not None
                and decode_result.avg_lorprob
                < logprob_threshold):
                needs_fallback = False  # silence
            if not needs_fallback:
                break
        
        return decode_result
    
    clip_idx = 0
    seek = seek_clips[clip_idx][0]
    # mel frames per output token: 2
    input_stride = exact_div(N_FRAMES,
                             model.dims.n_audio_ctx)
    time_precision = (input_stride
                      * HOP_LENGTH
                      / SAMPLE_RATE)  # time per output token: 0.02 (seconds)
    all_tokens = []
    all_segments = []
    prompt_reset_since = 0

    remaining_prompt_length = model.dims.n_text_ctx // 2 - 1
    if initial_prompt is not None:
        initial_prompt_tokens = tokensizer.encode(" "
                                                  + initial_prompt.strip())
        all_tokens.extend(initial_prompt_tokens)
        remaining_prompt_length -= len(initial_prompt_tokens)
    else:
        initial_prompt_tokens = []
    
    def new_segment(
            *,
            start: float,
            end: float,
            tokens: torch.Tensor,
            result: DecodingResult):
        
        tokens = tokens.tolist()
        text_tokens = [token for token in tokens
                       if token < tokensizer.eot]
        return {
            
        }