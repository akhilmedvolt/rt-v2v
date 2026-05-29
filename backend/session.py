"""Per-WebSocket session: owns queues, workers, interrupt state.

Each open WebSocket gets a fresh `Session`. The session wires the four
pipeline stages together with bounded `asyncio.Queue`s, exposes a tiny
`offer()` helper that drops the oldest item on overflow (we prefer fresh
audio to a backlog), and centralizes the interrupt protocol.

Interrupt protocol:
  1. `current_gen` is incremented so every in-flight queue item becomes stale.
  2. `interrupt_event` is set so workers racing against it (Whisper, GPT)
     short-circuit immediately.
  3. All downstream queues are drained.
  4. A `stop_playback` JSON event is sent to the client so it flushes its
     MediaSource buffer.
  5. The event is cleared so the next utterance can start cleanly.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from fastapi import WebSocket
from openai import AsyncOpenAI

from .config import Settings
from .messages import PlaybackItem, TranslationItem, TranscriptItem, UtteranceItem
from .workers.asr import asr_worker
from .workers.translator import translator_worker
from .workers.tts import tts_worker

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, websocket: WebSocket, settings: Settings) -> None:
        self.ws = websocket
        self.settings = settings

        self.audio_queue: asyncio.Queue[bytes] = asyncio.Queue(
            maxsize=settings.audio_queue_max
        )
        self.asr_queue: asyncio.Queue[UtteranceItem] = asyncio.Queue(
            maxsize=settings.asr_queue_max
        )
        self.translation_queue: asyncio.Queue[TranscriptItem] = asyncio.Queue(
            maxsize=settings.translation_queue_max
        )
        self.tts_queue: asyncio.Queue[TranslationItem] = asyncio.Queue(
            maxsize=settings.tts_queue_max
        )
        self.playback_queue: asyncio.Queue[PlaybackItem] = asyncio.Queue(
            maxsize=settings.playback_queue_max
        )

        self.interrupt_event = asyncio.Event()
        self.current_gen: int = 0

        self._send_lock = asyncio.Lock()
        self._first_chunk_seen: dict[int, bool] = {}
        self._tasks: list[asyncio.Task[Any]] = []
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)

    async def start(self) -> None:
        self._tasks.append(asyncio.create_task(asr_worker(self, self._openai), name="asr"))
        self._tasks.append(
            asyncio.create_task(translator_worker(self, self._openai), name="translator")
        )
        self._tasks.append(asyncio.create_task(tts_worker(self), name="tts"))
        self._tasks.append(asyncio.create_task(self._playback_dispatcher(), name="playback"))

    async def close(self) -> None:
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks.clear()
        try:
            await self._openai.close()
        except Exception:
            pass

    async def offer(self, queue: asyncio.Queue, item: Any) -> None:
        """Non-blocking put with drop-oldest on overflow.

        Latency matters more than completeness for a realtime translator: a
        full queue almost always means the user has moved on, so we'd rather
        evict stale items than block the producer.
        """

        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            try:
                dropped = queue.get_nowait()
                logger.warning(
                    "Queue full; dropped oldest item of type %s", type(dropped).__name__
                )
            except asyncio.QueueEmpty:
                pass
            try:
                queue.put_nowait(item)
            except asyncio.QueueFull:
                logger.error("Queue still full after eviction; dropping new item")

    async def send_event(self, payload: dict[str, Any]) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_json(payload)
            except Exception as exc:
                logger.debug("send_json failed: %s", exc)

    async def send_audio(self, mp3: bytes) -> None:
        async with self._send_lock:
            try:
                await self.ws.send_bytes(mp3)
            except Exception as exc:
                logger.debug("send_bytes failed: %s", exc)

    async def interrupt(self, reason: str = "speech_start") -> None:
        """Cancel the current turn end-to-end and flush downstream state."""

        prev_gen = self.current_gen
        self.current_gen += 1
        logger.info("Interrupting turn gen=%d (reason=%s)", prev_gen, reason)

        self.interrupt_event.set()
        _drain(self.translation_queue)
        _drain(self.tts_queue)
        _drain(self.playback_queue)
        self._first_chunk_seen.pop(prev_gen, None)

        await self.send_event({"type": "stop_playback", "reason": reason})
        await asyncio.sleep(0)
        self.interrupt_event.clear()

    def new_utterance(self, pcm: bytes, duration_ms: int) -> UtteranceItem:
        return UtteranceItem(
            pcm=pcm,
            duration_ms=duration_ms,
            gen=self.current_gen,
            t0=time.monotonic(),
        )

    async def _playback_dispatcher(self) -> None:
        """Forward MP3 chunks to the client; also reports latency on first chunk."""

        while True:
            item: PlaybackItem = await self.playback_queue.get()
            if item.gen != self.current_gen:
                logger.debug("Dropping stale playback chunk (gen=%d)", item.gen)
                continue

            if item.first_chunk and not self._first_chunk_seen.get(item.gen):
                self._first_chunk_seen[item.gen] = True
                tts_first_chunk_ms = (time.monotonic() - item.t0) * 1000.0 - (
                    item.asr_ms + item.translate_ms
                )
                total_ms = (time.monotonic() - item.t0) * 1000.0
                await self.send_event(
                    {
                        "type": "latency",
                        "asr_ms": round(item.asr_ms, 1),
                        "translate_ms": round(item.translate_ms, 1),
                        "tts_first_chunk_ms": round(max(0.0, tts_first_chunk_ms), 1),
                        "total_ms": round(total_ms, 1),
                    }
                )
                logger.info(
                    "Pipeline latency: asr=%.0fms translate=%.0fms tts_first=%.0fms total=%.0fms",
                    item.asr_ms,
                    item.translate_ms,
                    max(0.0, tts_first_chunk_ms),
                    total_ms,
                )

            await self.send_audio(item.mp3)


def _drain(queue: asyncio.Queue) -> int:
    dropped = 0
    while True:
        try:
            queue.get_nowait()
            dropped += 1
        except asyncio.QueueEmpty:
            break
    if dropped:
        logger.debug("Drained %d items from queue", dropped)
    return dropped
