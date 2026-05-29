"""Translator worker: English transcript -> Hindi text via GPT-4o-mini.

We make a single non-streaming chat completion per utterance. Streaming the
text wouldn't meaningfully reduce perceived latency because TTS still needs a
complete sentence to synthesize naturally, and non-streaming makes cancellation
trivial: just cancel the in-flight task.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from ..messages import TranscriptItem, TranslationItem
from ..prompts import build_messages
from .asr import _race_interrupt

if TYPE_CHECKING:
    from ..session import Session

logger = logging.getLogger(__name__)


async def translator_worker(session: "Session", client: AsyncOpenAI) -> None:
    settings = session.settings
    while True:
        item: TranscriptItem = await session.translation_queue.get()

        if item.gen != session.current_gen:
            logger.debug("Skipping stale translation item (gen=%d)", item.gen)
            continue

        t_tr_start = time.monotonic()

        async def _do_translate() -> str | None:
            try:
                resp = await client.chat.completions.create(
                    model=settings.translator_model,
                    messages=build_messages(item.english),
                    temperature=0.2,
                    max_tokens=400,
                )
            except Exception as exc:
                logger.exception("Translation failed: %s", exc)
                await session.send_event(
                    {"type": "error", "stage": "translate", "message": str(exc)}
                )
                return None
            choice = resp.choices[0].message.content or ""
            return choice.strip()

        hindi = await _race_interrupt(_do_translate(), session.interrupt_event)
        translate_ms = (time.monotonic() - t_tr_start) * 1000.0

        if hindi is None:
            logger.info("Translator interrupted or failed (gen=%d)", item.gen)
            continue
        if item.gen != session.current_gen:
            logger.debug("Translation completed but turn was interrupted")
            continue
        if not hindi:
            logger.info("Translator returned empty (pure filler?). Skipping TTS.")
            await session.send_event(
                {"type": "translation", "english": item.english, "hindi": ""}
            )
            continue

        logger.info("Translate (%.0f ms): %r -> %r", translate_ms, item.english, hindi)

        await session.send_event(
            {
                "type": "translation",
                "english": item.english,
                "hindi": hindi,
                "translate_ms": round(translate_ms, 1),
            }
        )

        out = TranslationItem(
            english=item.english,
            hindi=hindi,
            gen=item.gen,
            t0=item.t0,
            asr_ms=item.asr_ms,
            translate_ms=translate_ms,
        )
        await session.offer(session.tts_queue, out)
