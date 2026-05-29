"""ASR worker: utterance PCM -> English text via OpenAI Whisper.

Whisper's HTTP API is not a true streaming API, so we feed it whole utterances
that the VAD has already cut. We package PCM into an in-memory WAV (no ffmpeg
dependency) and race the API call against the session's interrupt event so a
new user utterance can short-circuit a slow inflight transcription.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import wave
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from ..messages import TranscriptItem, UtteranceItem

if TYPE_CHECKING:
    from ..session import Session

logger = logging.getLogger(__name__)


MIN_TRANSCRIPT_CHARS = 2


def _pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


async def _race_interrupt(coro, interrupt: asyncio.Event):
    """Run `coro` but cancel it as soon as the interrupt event fires.

    Returns the coro's result, or None if interrupted. If the caller itself is
    cancelled (e.g. the worker task is shutting down), the inner task is still
    cancelled and awaited so we never leak an orphaned API call that could
    later hit a closed HTTP client.
    """

    task = asyncio.create_task(coro)
    wait_task = asyncio.create_task(interrupt.wait())
    try:
        done, _ = await asyncio.wait(
            {task, wait_task}, return_when=asyncio.FIRST_COMPLETED
        )
        if task in done:
            return task.result()
        return None
    finally:
        for t in (task, wait_task):
            if not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass


async def asr_worker(session: "Session", client: AsyncOpenAI) -> None:
    settings = session.settings
    while True:
        item: UtteranceItem = await session.asr_queue.get()

        if item.gen != session.current_gen:
            logger.debug("Skipping stale ASR item (gen=%d)", item.gen)
            continue

        wav = _pcm_to_wav(item.pcm, settings.sample_rate)
        t_asr_start = time.monotonic()

        async def _do_transcribe() -> str | None:
            try:
                result = await client.audio.transcriptions.create(
                    model=settings.asr_model,
                    file=("utterance.wav", wav, "audio/wav"),
                    language="en",
                    response_format="text",
                    temperature=0.0,
                )
            except Exception as exc:
                if session.closing:
                    logger.debug("Whisper call aborted during shutdown: %s", exc)
                    return None
                logger.exception("Whisper transcription failed: %s", exc)
                await session.send_event(
                    {"type": "error", "stage": "asr", "message": str(exc)}
                )
                return None
            return (result if isinstance(result, str) else getattr(result, "text", "")).strip()

        text = await _race_interrupt(_do_transcribe(), session.interrupt_event)
        asr_ms = (time.monotonic() - t_asr_start) * 1000.0

        if text is None:
            logger.info("ASR interrupted or failed (gen=%d)", item.gen)
            continue
        if item.gen != session.current_gen:
            logger.debug("ASR completed but turn was interrupted")
            continue
        if len(text) < MIN_TRANSCRIPT_CHARS:
            logger.info("Dropping near-empty transcript: %r", text)
            continue

        logger.info("ASR (%.0f ms, %d ms audio): %r", asr_ms, item.duration_ms, text)

        await session.send_event(
            {"type": "partial_transcript", "text": text, "asr_ms": round(asr_ms, 1)}
        )

        transcript = TranscriptItem(
            english=text, gen=item.gen, t0=item.t0, asr_ms=asr_ms
        )
        await session.offer(session.translation_queue, transcript)
