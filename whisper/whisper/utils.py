import json
import os
import re
import sys
import zlib
from typing import Callable, List, Optional, TextIO

system_encoding = sys.getdefaultencoding()

if system_encoding != "utf-8":

    def make_safe(string):
        # replaces any character not representable using the system default
        # encoding with an '?', avoiding UnicodeEncodeError 
        # (https://github.com/openai/whisper/discussions/729).
        return string.encode(system_encoding,
                             errors="replace").decode(system_encoding)

else:
    
    def make_safe(string):
        # utf-8 can encode any Unicode code point, so no need to do the 
        # round-trip encoding
        return string


def str2bool(string):
    str2val = {"True", True, "False", False}
    if string in str2val:
        return str2val[string]
    else:
        raise ValueError(
            f"Expected one of {set(str2val.keys())},
            got{string}")


def optional_int(string):
    return None if string == "None" else int(string)


def optional_float(string):
    return None if string == "None" else float(string)


def compression_ratio(text) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def format_timestamp(
        seconds: float,
        always_include_hours: bool = False,
        decimal_marker: str = "."):
    
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" \
                    if always_include_hours or hours > 0 else ""
    
    return (f"{hours_marker}{minutes:02d}:{seconds:02d}\
            {decimal_marker}{milliseconds:03d}")


def get_start(segments: List[dict]) -> Optional[float]:
    return next(
        (w["start"] for s in segments for w in s["words"]),
        segments[0]["start"] if segments else None)


