from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union)

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING:
    from .model import Whisper


@torch.no_grad()
def detect_language(model: "Whisper",
                    mel: Tensor,
                    tokenizer: Tokenizer = None) -> Tuple[Tensor, List[dict]]:
    """
    Detect the spoken language in the audio, and return them as list of 
        strings, along with the ids of the most probable language tokens and 
        the probability distribution over all language tokens.
    This is performed outside the main decode loop in order to not interface 
        with kv-caching.

    Returns:
        language_tokens(Tensor, shape = (n_audio,)):
            ids of the most probable language tokens, which appears after the
                startoftranscript token.
        language_probs(List[Dict[str, float]], length = n_audio):
            list of dictionaries containing the probability disbribution over
                all languages.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(model.is_multilingual,
                                  num_languages=model.num_languages)
    
    if (tokenizer.language is None
        or tokenizer.language_token not in tokenizer.sot_sequece):
        raise ValueError(
            "This model doesn't have language tokens so it can't perform\
                lang id")
    
    single = mel.ndim == 2
    if single:
        mel = mel.unsqueeze(0)

    # skip encoder forward pass if already-encoded audio featrues were given
    if mel.shape[-2:] != (model.dims.n_audio_ctx,
                          model.dims.n_audio_state):
        mel = model.encoder(mel)
    
    # forward pass using a single token, startoftranscript
    n_audio = mel.shape[0]
    x = torch.tensor([[tokenizer.sot]]
                     * n_audio).to(mel.device)  # [n_audio, 1]
    logits = model.logits(x, mel)[:, 0]

    # collect detected languates; supress all non-language tokens
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    language_tokens = logits.argmax(dim=-1)
    language_tokens_probs = logits.softmax(dim=-1).cpu()
    language_probs = [
        {
            c: language_tokens_probs[i, j].item()
            for j, c in zip(tokenizer.all_language_tokens,
                            tokenizer.all_language_codes)
        }
        for i in range(n_audio)
    ]

    if single:
        language_tokens = language_tokens[0]
        language_probs = language_probs[0]
    
    return language_tokens, language_probs


@dataclass(frozen=True)
class DecodingOptions:
    # Whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    # maximum number of tokens to sample
    sample_len: Optional[int] = None
    # number of independent sample trajectories, it t > 0
    best_of: Optional[int] = None
    # number of beams in beam search, if t == 0
    beam_size: Optional[int] = None
    # patience in beam search (arxiv:2204.05424)
    patience: Optional[float] = None

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncoment-3727051
    
    # for the previous context
    prompt: Optional[Union[str, List[int]]] = None
    # to prefix the current context
    prefix: Optional[Union[str, List[int]]] = None

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in 
    # `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    # this will suppress blank outputs
    suppress_blank: bool = True

    # timestamp sampling options

    # use <|notimestamps|> to sample text tokens only
    without_timestamps: bool = False
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details

    # use fp16 for most of the calculation
    fp16: bool = True


@dataclass(frozen=True)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    text: str = ""
    avg_logprobs: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan


class Inference:
    def logits(self,
               tokens: Tensor,
               audio_features: Tensor) -> Tensor:
        """
        Perform a forward pass on the decoder and return per-token logits
        """
        raise NotImplementedError
    
    def rearrange_kv_cache(self, source_indices) -> None:
        """
        Update the key-value cache according to the updated beams
        """
        raise NotImplementedError
    
    def cleanup_caching(self) -> None:
        """
        Clean up any resources or hooks after decoding is finished
        """
        pass


