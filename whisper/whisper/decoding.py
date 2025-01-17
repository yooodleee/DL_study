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


class PyTorchInference(Inference):
    def __init__(
            self,
            model: "Whisper",
            initial_token_length: int):
        
        self.model = "Whisper" = model
        self.initial_token_length = initial_token_length
        self.kv_cache = {}
        self.hooks = []

        key_modules = [block.attn.key
                       for block in self.model.decoder.blocks]
        value_modules = [block.attn.value
                         for block in self.model.decoder.blocks]
        self.kv_modules = key_modules + value_modules
    
    def logits(
        self,
        tokens: Tensor,
        audio_features: Tensor) -> Tensor:

        if not self.kv_cache:
            self.kv_cache, self.hooks = self.model.install_kv_cache_hooks()
        
        if tokens.shape[-1] > self.initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]
        
        return self.model.decoder(tokens,
                                  audio_features,
                                  kv_cace=self.kv_cache)
    
    def cleanup_caching(self):
        for hook in self.hooks:
            hook.remove()
        
        self.kv_cache = {}
        self.hooks = []
    
    def rearrange_kv_cache(self, source_indices):
        if source_indices != list(range(len(source_indices))):
            for module in self.kv_modules:
                # update the key/value cache to contain the selected sequences
                self.kv_cache[module] = \
                    self.kv_cache[module][source_indices].detach()


class SequenceRanker:
    def rank(
            self,
            tokens: List[List[Tensor]],
            sum_logprobs: List[List[float]]) -> List[int]:
        """
        Given a list of groups of samples and their cumulative 
            log probabilities, return the indices of the samples in each
            group to select as the final result.
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using
        either a simple length normalization of Google NMT paper's length
        penalty.
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty
    
    def rank(
            self,
            tokens: List[List[Tensor]],
            sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result
        
        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        return [np.argmax(scores(p, l))
                for p, l in zip(sum_logprobs, lengths)]
    

class TokenDecoder:
    def reset(self):
        """
        Initialize any stateful variable for decoding a new sequence
        """
    
    def update(
            self,
            tokens: Tensor,
            logits: Tensor,
            sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        """
        Specify how to select the next token, based on the current 
            trace and logits.

        Parameters:
            tokens(Tensor, shape = (n_batch, current_sequence_length)):
                all tokens in the context so far, including the prefix and
                sot_sequence tokens
            logits(Tensor, shape = (n_batch, vocab_size)):
                per-token logits of the probability distribution at the 
                    current step
                sum_logprobs(Tensor, shape = (n_batch)):
                    cumulative log probabilities for each sequence

        Returns:
            tokens(Tensor, shape = (n_batch, current_sequence_length + 1)):
                the tokens, appended with the selected next token
            completed(bool):
                True if all sequences has reached the end of text
        """
        raise NotImplementedError
    
    def finalize(
            self,
            tokens: Tensor,
            sum_logprobs: Tensor) -> Tuple[Sequence[Sequence[Tensor]],
                                           List[List[float]]]:
        """
        Finalize search and return the final candidate sequences

        Parameters:
            tokens(Tensor, shape = (n_audio, n_group, 
            current_sequence_length)):
                all tokens in the context so far, including the prefix and 
                    sot sequence
            sum_logprobs(Tensor, shape = (n_audio, n_group)):
                cumulative log probabilities for each sequence
        
        Returns:
            tokens(Sequence[Sequence[Tensor]], length = n_audio):
                sequence of Tensors containing candidate token sequences,
                    for each audio input
            sum_logprobs(List[List[float]], length = n_audio):
                sequence of cumulative log probabilities corresponding 
                    to the above
        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot
    
    def update(
            self,
            tokens: Tensor,
            logits: Tensor,
            sum_logprobs: Tensor) -> Tuple[Tensor, bool]:
        
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits 
                                      / self.temperature).sample()
        
        logprobs = F.log_softmax(logits.float(), dim=-1)
        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), 
                                    next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed
    
    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


