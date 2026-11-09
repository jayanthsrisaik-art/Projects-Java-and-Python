"""Byte-level encoding detection.

Order of checks (first match wins):

1. Byte-order marks: UTF-8-SIG, UTF-16 LE/BE.
2. BOM-less UTF-16: a high share of NUL bytes on odd (LE) or even (BE) positions.
3. Strict UTF-8 decode. Valid multi-byte UTF-8 is very unlikely to happen by
   accident in a legacy single-byte file, so success is strong evidence.
4. cp1252 (Windows Western European, the usual "Excel export" encoding).
5. latin-1, which accepts every byte and is the last resort.
"""
from __future__ import annotations

import codecs
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingGuess:
    encoding: str  # codec name used to decode
    bom: bool
    reason: str


def detect_encoding(data: bytes) -> EncodingGuess:
    if data.startswith(codecs.BOM_UTF8):
        return EncodingGuess("utf-8-sig", True, "UTF-8 byte-order mark")
    if data.startswith(codecs.BOM_UTF16_LE) or data.startswith(codecs.BOM_UTF16_BE):
        return EncodingGuess("utf-16", True, "UTF-16 byte-order mark")

    sample = data[:65536]
    if len(sample) >= 4 and sample.count(0) > len(sample) // 4:
        odd_nuls = sample[1::2].count(0)
        even_nuls = sample[0::2].count(0)
        if odd_nuls > 0.9 * (len(sample) // 2):
            return EncodingGuess("utf-16-le", False, "NUL on odd bytes (BOM-less UTF-16LE)")
        if even_nuls > 0.9 * (len(sample) // 2):
            return EncodingGuess("utf-16-be", False, "NUL on even bytes (BOM-less UTF-16BE)")

    try:
        data.decode("utf-8")
        return EncodingGuess("utf-8", False, "decodes as strict UTF-8")
    except UnicodeDecodeError:
        pass
    try:
        data.decode("cp1252")
        return EncodingGuess("cp1252", False, "not valid UTF-8; decodes as cp1252")
    except UnicodeDecodeError:
        return EncodingGuess("latin-1", False, "not valid UTF-8 or cp1252; latin-1 fallback")


def decode(data: bytes, guess: EncodingGuess | None = None) -> tuple[str, EncodingGuess]:
    guess = guess or detect_encoding(data)
    text = data.decode(guess.encoding)
    if text.startswith("﻿"):  # utf-16 codec keeps no BOM, but be defensive
        text = text[1:]
    return text, guess
