"""VAD-driven utterance segmenter.

We can't stream raw audio into Whisper, so we use a simple webrtcvad-based
state machine to slice the live PCM stream into utterances:

    silence -> speech_start -> ... voiced frames ... -> trailing silence -> utterance

A small ring of pre-speech frames is kept so we don't clip the leading
consonant of an utterance, and short bursts (< min_speech_ms) are dropped
to suppress clicks and pops.

The segmenter exposes a tiny synchronous `feed(pcm)` API that returns a list
of events. The caller (the websocket consumer task) owns the asyncio glue.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import webrtcvad

logger = logging.getLogger(__name__)


FRAME_MS = 20  # webrtcvad supports 10/20/30 ms; 20 ms is the standard sweet spot
BYTES_PER_SAMPLE = 2  # Int16 mono


@dataclass
class SpeechStart:
    """Emitted the moment voiced audio is detected after a silence stretch."""


@dataclass
class Utterance:
    """Emitted once a trailing-silence threshold confirms end-of-utterance."""

    pcm: bytes
    duration_ms: int


VadEvent = SpeechStart | Utterance


class UtteranceSegmenter:
    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 2,
        silence_ms: int = 500,
        min_speech_ms: int = 300,
        pre_speech_ms: int = 200,
    ) -> None:
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(
                f"webrtcvad requires sample_rate in 8000/16000/32000/48000, got {sample_rate}"
            )

        self._sr = sample_rate
        self._frame_bytes = int(sample_rate * FRAME_MS / 1000) * BYTES_PER_SAMPLE
        self._silence_frames_needed = max(1, silence_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._pre_speech_frames = max(0, pre_speech_ms // FRAME_MS)

        self._vad = webrtcvad.Vad(aggressiveness)

        self._leftover = bytearray()
        self._pre_buffer: deque[bytes] = deque(maxlen=self._pre_speech_frames)
        self._utterance: bytearray = bytearray()
        self._in_speech = False
        self._trailing_silence_frames = 0
        self._voiced_frames_in_utterance = 0

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def feed(self, pcm: bytes) -> list[VadEvent]:
        events: list[VadEvent] = []
        if not pcm:
            return events

        self._leftover.extend(pcm)

        while len(self._leftover) >= self._frame_bytes:
            frame = bytes(self._leftover[: self._frame_bytes])
            del self._leftover[: self._frame_bytes]
            self._process_frame(frame, events)

        return events

    def reset(self) -> None:
        """Drop any in-flight utterance state. Called on interruption."""

        self._leftover.clear()
        self._pre_buffer.clear()
        self._utterance.clear()
        self._in_speech = False
        self._trailing_silence_frames = 0
        self._voiced_frames_in_utterance = 0

    def _process_frame(self, frame: bytes, events: list[VadEvent]) -> None:
        is_speech = self._vad.is_speech(frame, self._sr)

        if not self._in_speech:
            self._pre_buffer.append(frame)
            if is_speech:
                self._in_speech = True
                self._trailing_silence_frames = 0
                self._voiced_frames_in_utterance = 1
                self._utterance.clear()
                for buffered in self._pre_buffer:
                    self._utterance.extend(buffered)
                self._pre_buffer.clear()
                events.append(SpeechStart())
            return

        self._utterance.extend(frame)
        if is_speech:
            self._trailing_silence_frames = 0
            self._voiced_frames_in_utterance += 1
            return

        self._trailing_silence_frames += 1
        if self._trailing_silence_frames < self._silence_frames_needed:
            return

        if self._voiced_frames_in_utterance >= self._min_speech_frames:
            duration_ms = (
                len(self._utterance) // BYTES_PER_SAMPLE * 1000 // self._sr
            )
            events.append(Utterance(pcm=bytes(self._utterance), duration_ms=duration_ms))
        else:
            logger.debug(
                "Discarded short utterance (%d voiced frames < %d)",
                self._voiced_frames_in_utterance,
                self._min_speech_frames,
            )

        self._utterance.clear()
        self._in_speech = False
        self._trailing_silence_frames = 0
        self._voiced_frames_in_utterance = 0


def chunk_iter(pcm: bytes, frame_bytes: int) -> Iterable[bytes]:
    """Tiny helper for tests: split a buffer into fixed-size frames."""

    for i in range(0, len(pcm) - frame_bytes + 1, frame_bytes):
        yield pcm[i : i + frame_bytes]
