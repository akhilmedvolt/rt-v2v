"""Lightweight data classes that flow between pipeline stages.

Every item carries a `gen` (generation) tag so workers can cheaply drop work
that belongs to an interrupted turn, and a `t0` monotonic timestamp so we can
report end-to-end latency to the client.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class UtteranceItem:
    """Finalized speech segment ready for ASR."""

    pcm: bytes
    duration_ms: int
    gen: int
    t0: float = field(default_factory=time.monotonic)


@dataclass
class TranscriptItem:
    """Stabilized English transcript ready for translation."""

    english: str
    gen: int
    t0: float
    asr_ms: float


@dataclass
class TranslationItem:
    """Hindi translation ready for TTS."""

    english: str
    hindi: str
    gen: int
    t0: float
    asr_ms: float
    translate_ms: float


@dataclass
class PlaybackItem:
    """A single MP3 chunk to forward to the client.

    `first_chunk` marks the first chunk of an utterance so we can stamp
    `tts_first_chunk_ms` exactly once.
    """

    mp3: bytes
    gen: int
    first_chunk: bool
    t0: float
    asr_ms: float
    translate_ms: float
