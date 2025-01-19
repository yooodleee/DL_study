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
    