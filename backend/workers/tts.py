"""TTS worker: Hindi text -> streamed MP3 chunks via edge-tts.

edge-tts is free, has solid Hindi voices (`hi-IN-MadhurNeural`,
`hi-IN-SwaraNeural`), and natively streams MP3 chunks as the synthesis runs.
We forward each chunk to the playback queue and bail out the moment the
session's interrupt event fires.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

import edge_tts

from ..messages import PlaybackItem, TranslationItem

if TYPE_CHECKING:
    from ..session import Session

logger = logging.getLogger(__name__)


async def tts_worker(session: "Session") -> None:
    settings = session.settings
    while True:
        item: TranslationItem = await session.tts_queue.get()

        if item.gen != session.current_gen:
            logger.debug("Skipping stale TTS item (gen=%d)", item.gen)
            continue

        first_chunk_emitted = False
        t_tts_start = time.monotonic()

        try:
            communicate = edge_tts.Communicate(item.hindi, settings.tts_voice)
            async for chunk in communicate.stream():
                if session.interrupt_event.is_set():
                    logger.info("TTS interrupted mid-stream (gen=%d)", item.gen)
                    break
                if item.gen != session.current_gen:
                    logger.debug("TTS turn rotated mid-stream")
                    break
                if chunk.get("type") != "audio":
                    continue
                data: bytes = chunk["data"]
                if not data:
                    continue

                playback = PlaybackItem(
                    mp3=data,
                    gen=item.gen,
                    first_chunk=not first_chunk_emitted,
                    t0=item.t0,
                    asr_ms=item.asr_ms,
                    translate_ms=item.translate_ms,
                )
                if not first_chunk_emitted:
                    first_chunk_ms = (time.monotonic() - t_tts_start) * 1000.0
                    logger.info(
                        "TTS first chunk at %.0f ms (gen=%d)", first_chunk_ms, item.gen
                    )
                    first_chunk_emitted = True
                await session.offer(session.playback_queue, playback)
        except Exception as exc:
            if session.closing:
                logger.debug("TTS aborted during shutdown: %s", exc)
                continue
            logger.exception("TTS synthesis failed: %s", exc)
            await session.send_event(
                {"type": "error", "stage": "tts", "message": str(exc)}
            )
