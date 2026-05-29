"""Centralized configuration loaded from environment variables.

Keeping this in one place makes the tunables easy to discover and tweak
without grepping through worker code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    openai_api_key: str

    asr_model: str
    translator_model: str
    tts_voice: str

    sample_rate: int

    vad_aggressiveness: int
    vad_silence_ms: int
    vad_min_speech_ms: int

    audio_queue_max: int
    asr_queue_max: int
    translation_queue_max: int
    tts_queue_max: int
    playback_queue_max: int

    log_level: str


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        asr_model=os.getenv("ASR_MODEL", "whisper-1"),
        translator_model=os.getenv("TRANSLATOR_MODEL", "gpt-4o-mini"),
        tts_voice=os.getenv("TTS_VOICE", "hi-IN-MadhurNeural"),
        sample_rate=_int("SAMPLE_RATE", 16000),
        vad_aggressiveness=_int("VAD_AGGRESSIVENESS", 2),
        vad_silence_ms=_int("VAD_SILENCE_MS", 500),
        vad_min_speech_ms=_int("VAD_MIN_SPEECH_MS", 300),
        audio_queue_max=_int("AUDIO_QUEUE_MAX", 200),
        asr_queue_max=_int("ASR_QUEUE_MAX", 8),
        translation_queue_max=_int("TRANSLATION_QUEUE_MAX", 8),
        tts_queue_max=_int("TTS_QUEUE_MAX", 8),
        playback_queue_max=_int("PLAYBACK_QUEUE_MAX", 64),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )


settings = load_settings()


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
    )
