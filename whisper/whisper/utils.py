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


